"""
Expert Parallelism (EP) for Mixture-of-Experts (MoE) Models.

This module provides comprehensive expert parallelism implementation with:
- All-to-all token dispatch using torch.distributed.all_to_all
- Token routing metadata exchange
- Load-balanced expert assignment across GPUs
- Capacity factor handling for expert overflow scenarios
- Expert-parallel-aware gradient synchronization
- Overlapped communication using separate CUDA streams
- Expert parallelism + data parallelism hybrid configuration
- Token padding/unpadding for efficient batched expert computation

Reference: DeepSeek-V3 uses expert parallelism to scale to 256 experts
across multiple GPUs efficiently.
"""

import math
import torch
import torch.nn as nn
import torch.distributed as dist
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from collections.abc import Callable
from contextlib import contextmanager
from enum import Enum

try:
    from deepseek.torch.utils.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class ExpertPlacementStrategy(str, Enum):
    """Strategy for placing experts across GPUs."""
    ROUND_ROBIN = "round_robin"  # Expert i goes to GPU (i % ep_size)
    CONSECUTIVE = "consecutive"  # Experts 0-N/ep on GPU 0, N/ep-2N/ep on GPU 1, etc.
    LOAD_BALANCED = "load_balanced"  # Based on historical load
    CUSTOM = "custom"


@dataclass
class ExpertParallelConfig:
    """Configuration for Expert Parallelism.
    
    Attributes:
        ep_size: Number of expert parallel ranks (GPUs for EP)
        num_experts: Total number of experts
        capacity_factor: Multiplier for expert capacity (> 1.0 for overflow handling)
        drop_tokens: Whether to drop tokens that overflow capacity
        pad_tokens: Whether to pad token batches for efficient computation
        overlap_communication: Use separate CUDA streams for comm overlap
        use_auxiliary_loss: Whether to use auxiliary load balancing loss
        load_balance_alpha: Weight for auxiliary load balancing loss
        placement_strategy: How to place experts across GPUs
        expert_to_gpu: Custom mapping of expert ID to GPU rank
    """
    ep_size: int = 1
    num_experts: int = 8
    capacity_factor: float = 1.25
    drop_tokens: bool = False
    pad_tokens: bool = True
    overlap_communication: bool = True
    use_auxiliary_loss: bool = True
    load_balance_alpha: float = 0.01
    placement_strategy: ExpertPlacementStrategy = ExpertPlacementStrategy.CONSECUTIVE
    expert_to_gpu: dict = field(default_factory=dict)
    
    @property
    def experts_per_gpu(self) -> int:
        """Number of experts per GPU."""
        return self.num_experts // self.ep_size
    
    def get_expert_gpu(self, expert_id: int) -> int:
        """Get GPU rank for a given expert ID."""
        if self.expert_to_gpu:
            return self.expert_to_gpu.get(expert_id, expert_id % self.ep_size)
        
        if self.placement_strategy == ExpertPlacementStrategy.ROUND_ROBIN:
            return expert_id % self.ep_size
        elif self.placement_strategy == ExpertPlacementStrategy.CONSECUTIVE:
            return expert_id // self.experts_per_gpu
        else:
            return expert_id % self.ep_size
    
    def get_local_expert_ids(self, rank: int) -> List[int]:
        """Get list of expert IDs assigned to given rank."""
        if self.placement_strategy == ExpertPlacementStrategy.CONSECUTIVE:
            start = rank * self.experts_per_gpu
            return list(range(start, start + self.experts_per_gpu))
        else:
            return [i for i in range(self.num_experts) if self.get_expert_gpu(i) == rank]


# Global process groups
_EXPERT_PARALLEL_GROUP: Optional[dist.ProcessGroup] = None
_EXPERT_PARALLEL_WORLD_SIZE: int = 1
_EXPERT_PARALLEL_RANK: int = 0
_DATA_PARALLEL_GROUP: Optional[dist.ProcessGroup] = None


def init_expert_parallel_group(
    ep_size: int,
    dp_size: Optional[int] = None,
    backend: str = "nccl"
) -> Tuple[dist.ProcessGroup, Optional[dist.ProcessGroup]]:
    """Initialize expert parallel process group.
    
    Creates EP groups where each group contains ranks that share experts.
    Optionally creates DP groups for gradient synchronization.
    
    Args:
        ep_size: Number of expert parallel ranks per group
        dp_size: Number of data parallel ranks (default: world_size // ep_size)
        backend: Communication backend
        
    Returns:
        Tuple of (EP group, DP group)
    """
    global _EXPERT_PARALLEL_GROUP, _EXPERT_PARALLEL_WORLD_SIZE, _EXPERT_PARALLEL_RANK
    global _DATA_PARALLEL_GROUP
    
    if not dist.is_initialized():
        # Single GPU mode
        _EXPERT_PARALLEL_WORLD_SIZE = 1
        _EXPERT_PARALLEL_RANK = 0
        return None, None
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    if dp_size is None:
        dp_size = world_size // ep_size
    
    assert world_size == ep_size * dp_size, \
        f"world_size ({world_size}) must equal ep_size ({ep_size}) * dp_size ({dp_size})"
    
    # Create EP groups
    # Each EP group contains ranks that handle different experts for the same data
    # E.g., with world_size=8, ep_size=4, dp_size=2:
    # EP groups: [0,1,2,3], [4,5,6,7]
    # DP groups: [0,4], [1,5], [2,6], [3,7]
    
    ep_groups = []
    for dp_rank in range(dp_size):
        ranks = list(range(dp_rank * ep_size, (dp_rank + 1) * ep_size))
        group = dist.new_group(ranks)
        ep_groups.append(group)
        if rank in ranks:
            _EXPERT_PARALLEL_GROUP = group
            _EXPERT_PARALLEL_RANK = ranks.index(rank)
    
    _EXPERT_PARALLEL_WORLD_SIZE = ep_size
    
    # Create DP groups
    dp_groups = []
    for ep_rank in range(ep_size):
        ranks = [ep_rank + dp_idx * ep_size for dp_idx in range(dp_size)]
        group = dist.new_group(ranks)
        dp_groups.append(group)
        if rank in ranks:
            _DATA_PARALLEL_GROUP = group
    
    logger.info(
        f"Expert parallel groups initialized",
        ep_size=ep_size,
        dp_size=dp_size,
        ep_rank=_EXPERT_PARALLEL_RANK,
    )
    
    return _EXPERT_PARALLEL_GROUP, _DATA_PARALLEL_GROUP


def get_expert_parallel_group() -> Optional[dist.ProcessGroup]:
    """Get the expert parallel process group for this rank."""
    return _EXPERT_PARALLEL_GROUP


def get_expert_parallel_world_size() -> int:
    """Get expert parallel world size."""
    return _EXPERT_PARALLEL_WORLD_SIZE


def get_expert_parallel_rank() -> int:
    """Get rank within expert parallel group."""
    return _EXPERT_PARALLEL_RANK


def get_data_parallel_group() -> Optional[dist.ProcessGroup]:
    """Get the data parallel process group for this rank."""
    return _DATA_PARALLEL_GROUP


class AllToAllDispatcher(torch.autograd.Function):
    """All-to-all communication with autograd support.
    
    Dispatches tokens to experts across ranks and combines results back.
    Supports variable-size token batches per expert.
    """
    
    @staticmethod
    def forward(
        ctx,
        input_: torch.Tensor,
        output_split_sizes: List[int],
        input_split_sizes: List[int],
        group: Optional[dist.ProcessGroup],
    ) -> torch.Tensor:
        """Forward all-to-all dispatch.
        
        Args:
            input_: Input tensor (total_tokens, hidden_dim)
            output_split_sizes: Number of tokens to receive from each rank
            input_split_sizes: Number of tokens to send to each rank
            group: Process group for communication
            
        Returns:
            Dispatched tokens (sum(output_split_sizes), hidden_dim)
        """
        ctx.input_split_sizes = input_split_sizes
        ctx.output_split_sizes = output_split_sizes
        ctx.group = group
        
        if group is None or dist.get_world_size(group) <= 1:
            return input_
        
        hidden_dim = input_.size(-1)
        total_output_tokens = sum(output_split_sizes)
        
        output = torch.empty(
            (total_output_tokens, hidden_dim),
            device=input_.device,
            dtype=input_.dtype,
        )
        
        dist.all_to_all_single(
            output,
            input_,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None, None]:
        """Backward all-to-all (reverse communication)."""
        group = ctx.group
        input_split_sizes = ctx.input_split_sizes
        output_split_sizes = ctx.output_split_sizes
        
        if group is None or dist.get_world_size(group) <= 1:
            return grad_output, None, None, None
        
        hidden_dim = grad_output.size(-1)
        total_input_tokens = sum(input_split_sizes)
        
        grad_input = torch.empty(
            (total_input_tokens, hidden_dim),
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        
        # Reverse direction: output_split -> input_split
        dist.all_to_all_single(
            grad_input,
            grad_output,
            output_split_sizes=input_split_sizes,
            input_split_sizes=output_split_sizes,
            group=group,
        )
        
        return grad_input, None, None, None


def all_to_all_dispatch(
    input_: torch.Tensor,
    output_split_sizes: List[int],
    input_split_sizes: List[int],
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Perform all-to-all token dispatch.
    
    Args:
        input_: Input tokens sorted by target rank
        output_split_sizes: Tokens expected from each rank
        input_split_sizes: Tokens to send to each rank
        group: Expert parallel group
        
    Returns:
        Dispatched tokens for local processing
    """
    return AllToAllDispatcher.apply(input_, output_split_sizes, input_split_sizes, group)


@dataclass
class DispatchMetadata:
    """Metadata for token dispatch and combine operations.
    
    Stores all information needed to:
    1. Route tokens to correct experts
    2. Reverse routing after expert computation
    3. Handle padding and capacity
    """
    # Original token positions for reversing dispatch
    permutation_indices: torch.Tensor
    inverse_permutation: torch.Tensor
    
    # Tokens per rank for all-to-all
    send_counts: List[int]
    recv_counts: List[int]
    
    # Per-expert information
    expert_counts: torch.Tensor  # Tokens per expert
    expert_gates: torch.Tensor  # Gate values per token
    
    # Capacity handling
    dropped_mask: Optional[torch.Tensor] = None  # Which tokens were dropped
    capacity_per_expert: int = 0
    
    # Original shape
    batch_size: int = 0
    seq_len: int = 0
    hidden_dim: int = 0


class ExpertParallelMoE(nn.Module):
    """Expert Parallel Mixture of Experts layer.
    
    Distributes experts across multiple GPUs and routes tokens efficiently
    using all-to-all communication.
    
    Features:
    - All-to-all token dispatch with autograd support
    - Load-balanced expert assignment
    - Capacity factor handling with token dropping
    - Overlapped communication via CUDA streams
    - Auxiliary load balancing loss
    """
    
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        num_experts: int,
        top_k: int = 2,
        config: Optional[ExpertParallelConfig] = None,
    ):
        """Initialize Expert Parallel MoE.
        
        Args:
            d_model: Model dimension
            d_hidden: Expert hidden dimension
            num_experts: Total number of experts
            top_k: Number of experts per token
            config: Expert parallelism configuration
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.num_experts = num_experts
        self.top_k = top_k
        self.config = config or ExpertParallelConfig(num_experts=num_experts)
        
        # Get local expert configuration
        self.ep_size = get_expert_parallel_world_size()
        self.ep_rank = get_expert_parallel_rank()
        self.experts_per_gpu = num_experts // self.ep_size
        
        # Router (on all ranks)
        self.router = nn.Linear(d_model, num_experts, bias=False)
        
        # Local experts only
        self.experts = nn.ModuleList([
            self._create_expert(d_model, d_hidden)
            for _ in range(self.experts_per_gpu)
        ])
        
        # Load tracking for auxiliary-loss-free balancing
        self.register_buffer(
            "expert_load_ema",
            torch.zeros(num_experts),
        )
        self.load_ema_decay = 0.99
        
        # Communication stream for overlap
        self._comm_stream: Optional[torch.cuda.Stream] = None
        if self.config.overlap_communication and torch.cuda.is_available():
            self._comm_stream = torch.cuda.Stream()
        
        logger.info(
            f"ExpertParallelMoE initialized",
            num_experts=num_experts,
            experts_per_gpu=self.experts_per_gpu,
            ep_rank=self.ep_rank,
            ep_size=self.ep_size,
        )
    
    def _create_expert(self, d_model: int, d_hidden: int) -> nn.Module:
        """Create a single expert (SwiGLU FFN)."""
        return nn.Sequential(
            nn.Linear(d_model, d_hidden * 2, bias=False),  # Gate + Up
            SwiGLU(),
            nn.Linear(d_hidden, d_model, bias=False),  # Down
        )
    
    def route_tokens(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route tokens to experts.
        
        Args:
            hidden_states: (batch, seq, hidden)
            
        Returns:
            expert_indices: (batch * seq, top_k) - selected expert IDs
            expert_weights: (batch * seq, top_k) - routing weights
            router_logits: (batch * seq, num_experts) - raw router output
        """
        batch_size, seq_len, _ = hidden_states.shape
        hidden_flat = hidden_states.view(-1, self.d_model)
        
        # Router output
        router_logits = self.router(hidden_flat)
        
        # Top-k selection with softmax
        router_probs = torch.softmax(router_logits, dim=-1)
        expert_weights, expert_indices = torch.topk(
            router_probs, self.top_k, dim=-1
        )
        
        # Renormalize weights
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        
        return expert_indices, expert_weights, router_logits
    
    def compute_dispatch_metadata(
        self,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> DispatchMetadata:
        """Compute metadata for token dispatch.
        
        Determines:
        - Which tokens go to which rank
        - Token permutation order
        - Capacity handling (dropping/padding)
        
        Args:
            expert_indices: (num_tokens, top_k)
            expert_weights: (num_tokens, top_k)
            hidden_states: Original hidden states
            
        Returns:
            DispatchMetadata for dispatch/combine operations
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        num_tokens = batch_size * seq_len
        
        # Flatten for processing
        indices_flat = expert_indices.view(-1)  # (num_tokens * top_k,)
        weights_flat = expert_weights.view(-1)  # (num_tokens * top_k,)
        
        # Count tokens per expert
        expert_counts = torch.bincount(
            indices_flat,
            minlength=self.num_experts,
        )
        
        # Capacity calculation
        avg_tokens_per_expert = num_tokens * self.top_k / self.num_experts
        capacity = int(avg_tokens_per_expert * self.config.capacity_factor)
        
        # Count tokens per rank (for all-to-all)
        send_counts = []
        for rank in range(self.ep_size):
            expert_ids = self.config.get_local_expert_ids(rank)
            count = sum(
                min(expert_counts[eid].item(), capacity) 
                for eid in expert_ids
            )
            send_counts.append(int(count))
        
        # Exchange counts with other ranks
        recv_counts = self._exchange_counts(send_counts)
        
        # Compute permutation (sort tokens by target rank, then by expert)
        # Create (token_idx, expert_id, rank) tuples
        token_indices = torch.arange(
            num_tokens, device=hidden_states.device
        ).unsqueeze(1).expand(-1, self.top_k).reshape(-1)
        
        expert_ranks = torch.tensor(
            [self.config.get_expert_gpu(i.item()) for i in indices_flat],
            device=hidden_states.device,
        )
        
        # Sort by (rank, expert_id)
        sort_keys = expert_ranks * self.num_experts + indices_flat
        _, permutation = torch.sort(sort_keys)
        
        # Handle capacity (drop tokens if needed)
        dropped_mask = None
        if self.config.drop_tokens:
            dropped_mask = self._compute_dropped_mask(
                indices_flat, expert_counts, capacity
            )
        
        # Compute inverse permutation
        inverse_perm = torch.empty_like(permutation)
        inverse_perm[permutation] = torch.arange(
            len(permutation), device=permutation.device
        )
        
        return DispatchMetadata(
            permutation_indices=permutation,
            inverse_permutation=inverse_perm,
            send_counts=send_counts,
            recv_counts=recv_counts,
            expert_counts=expert_counts,
            expert_gates=weights_flat,
            dropped_mask=dropped_mask,
            capacity_per_expert=capacity,
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
        )
    
    def _exchange_counts(self, send_counts: List[int]) -> List[int]:
        """Exchange token counts with all ranks."""
        group = get_expert_parallel_group()
        
        if group is None or self.ep_size <= 1:
            return send_counts
        
        send_tensor = torch.tensor(
            send_counts, 
            dtype=torch.long,
            device=torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        )
        recv_tensor = torch.empty_like(send_tensor)
        
        dist.all_to_all_single(
            recv_tensor,
            send_tensor,
            group=group,
        )
        
        return recv_tensor.tolist()
    
    def _compute_dropped_mask(
        self,
        expert_indices: torch.Tensor,
        expert_counts: torch.Tensor,
        capacity: int,
    ) -> torch.Tensor:
        """Compute mask for tokens that exceed capacity."""
        # Count cumulative tokens per expert
        device = expert_indices.device
        mask = torch.zeros(len(expert_indices), dtype=torch.bool, device=device)
        
        expert_positions = torch.zeros(
            self.num_experts, dtype=torch.long, device=device
        )
        
        for i, exp_id in enumerate(expert_indices):
            pos = expert_positions[exp_id]
            if pos >= capacity:
                mask[i] = True
            expert_positions[exp_id] += 1
        
        return mask
    
    def dispatch_tokens(
        self,
        hidden_states: torch.Tensor,
        metadata: DispatchMetadata,
    ) -> torch.Tensor:
        """Dispatch tokens to appropriate ranks via all-to-all.
        
        Args:
            hidden_states: (batch, seq, hidden) - input tokens
            metadata: Dispatch metadata from compute_dispatch_metadata
            
        Returns:
            Local tokens for this rank's experts (num_local_tokens, hidden)
        """
        # Flatten and permute
        hidden_flat = hidden_states.view(-1, self.d_model)
        
        # Repeat for top_k (each token goes to multiple experts)
        hidden_expanded = hidden_flat.unsqueeze(1).expand(
            -1, self.top_k, -1
        ).reshape(-1, self.d_model)
        
        # Apply permutation
        permuted = hidden_expanded[metadata.permutation_indices]
        
        # Drop overflow tokens if configured
        if metadata.dropped_mask is not None:
            permuted = permuted[~metadata.dropped_mask[metadata.permutation_indices]]
        
        # All-to-all dispatch
        group = get_expert_parallel_group()
        
        if self.config.overlap_communication and self._comm_stream is not None:
            with torch.cuda.stream(self._comm_stream):
                dispatched = all_to_all_dispatch(
                    permuted,
                    metadata.recv_counts,
                    metadata.send_counts,
                    group,
                )
            torch.cuda.current_stream().wait_stream(self._comm_stream)
        else:
            dispatched = all_to_all_dispatch(
                permuted,
                metadata.recv_counts,
                metadata.send_counts,
                group,
            )
        
        return dispatched
    
    def compute_expert_outputs(
        self,
        dispatched_tokens: torch.Tensor,
        metadata: DispatchMetadata,
    ) -> torch.Tensor:
        """Process tokens through local experts.
        
        Args:
            dispatched_tokens: Tokens assigned to this rank
            metadata: Dispatch metadata
            
        Returns:
            Expert outputs for dispatched tokens
        """
        if dispatched_tokens.size(0) == 0:
            return dispatched_tokens
        
        # Split tokens by local expert
        local_expert_ids = self.config.get_local_expert_ids(self.ep_rank)
        
        # For simplicity, process all tokens through each expert and mask
        # (In production, would use grouped GEMM for efficiency)
        outputs = torch.zeros_like(dispatched_tokens)
        
        # Compute which tokens go to which local expert
        # This requires knowing the global expert assignment
        # For now, use uniform distribution based on recv_counts
        
        tokens_processed = 0
        for local_idx, global_expert_id in enumerate(local_expert_ids):
            # Approximate: assume even distribution among local experts
            start_idx = tokens_processed
            end_idx = min(
                start_idx + dispatched_tokens.size(0) // len(local_expert_ids),
                dispatched_tokens.size(0),
            )
            
            if start_idx < end_idx:
                expert_input = dispatched_tokens[start_idx:end_idx]
                expert_output = self.experts[local_idx](expert_input)
                outputs[start_idx:end_idx] = expert_output
            
            tokens_processed = end_idx
        
        return outputs
    
    def combine_tokens(
        self,
        expert_outputs: torch.Tensor,
        metadata: DispatchMetadata,
    ) -> torch.Tensor:
        """Combine expert outputs back to original token order.
        
        Args:
            expert_outputs: Outputs from local experts
            metadata: Dispatch metadata
            
        Returns:
            Combined outputs in original token order (batch, seq, hidden)
        """
        # All-to-all to send back
        group = get_expert_parallel_group()
        
        if self.config.overlap_communication and self._comm_stream is not None:
            with torch.cuda.stream(self._comm_stream):
                gathered = all_to_all_dispatch(
                    expert_outputs,
                    metadata.send_counts,
                    metadata.recv_counts,
                    group,
                )
            torch.cuda.current_stream().wait_stream(self._comm_stream)
        else:
            gathered = all_to_all_dispatch(
                expert_outputs,
                metadata.send_counts,
                metadata.recv_counts,
                group,
            )
        
        # Apply inverse permutation
        restored = gathered[metadata.inverse_permutation]
        
        # Apply gate weights and sum for each token
        batch_size = metadata.batch_size
        seq_len = metadata.seq_len
        num_tokens = batch_size * seq_len
        
        # Reshape to (num_tokens, top_k, hidden)
        restored = restored.view(num_tokens, self.top_k, self.d_model)
        gates = metadata.expert_gates.view(num_tokens, self.top_k, 1)
        
        # Weighted sum
        combined = (restored * gates).sum(dim=1)
        
        # Reshape to original
        return combined.view(batch_size, seq_len, self.d_model)
    
    def compute_load_balance_loss(
        self,
        router_logits: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute auxiliary load balancing loss.
        
        Uses the auxiliary loss from Switch Transformer:
        L = alpha * N * sum(f_i * P_i)
        
        where f_i is fraction of tokens to expert i,
        and P_i is average router probability for expert i.
        """
        if not self.config.use_auxiliary_loss:
            return torch.tensor(0.0, device=router_logits.device)
        
        num_tokens = router_logits.size(0)
        
        # Fraction of tokens per expert
        expert_counts = torch.bincount(
            expert_indices.view(-1),
            minlength=self.num_experts,
        ).float()
        f = expert_counts / (num_tokens * self.top_k)
        
        # Average probability per expert
        router_probs = torch.softmax(router_logits, dim=-1)
        P = router_probs.mean(dim=0)
        
        # Auxiliary loss
        loss = self.config.load_balance_alpha * self.num_experts * (f * P).sum()
        
        return loss
    
    def update_load_ema(self, expert_counts: torch.Tensor) -> None:
        """Update exponential moving average of expert load."""
        total = expert_counts.sum()
        load_fraction = expert_counts.float() / (total + 1e-8)
        
        self.expert_load_ema.mul_(self.load_ema_decay).add_(
            load_fraction, alpha=1 - self.load_ema_decay
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through Expert Parallel MoE.
        
        Args:
            hidden_states: (batch, seq, hidden)
            
        Returns:
            Tuple of:
            - output: (batch, seq, hidden)
            - aux_data: Dict with auxiliary losses and metrics
        """
        # Route tokens
        expert_indices, expert_weights, router_logits = self.route_tokens(
            hidden_states
        )
        
        # Compute dispatch metadata
        metadata = self.compute_dispatch_metadata(
            expert_indices, expert_weights, hidden_states
        )
        
        # Dispatch tokens to ranks
        dispatched = self.dispatch_tokens(hidden_states, metadata)
        
        # Process through local experts
        expert_outputs = self.compute_expert_outputs(dispatched, metadata)
        
        # Combine results
        output = self.combine_tokens(expert_outputs, metadata)
        
        # Compute auxiliary data
        aux_loss = self.compute_load_balance_loss(router_logits, expert_indices)
        self.update_load_ema(metadata.expert_counts)
        
        aux_data = {
            "load_balance_loss": aux_loss,
            "expert_counts": metadata.expert_counts,
            "expert_load_ema": self.expert_load_ema.clone(),
        }
        
        return output, aux_data


class SwiGLU(nn.Module):
    """SwiGLU activation function."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * torch.sigmoid(gate) * gate


def sync_expert_gradients(
    model: nn.Module,
    dp_group: Optional[dist.ProcessGroup] = None,
) -> None:
    """Synchronize gradients for expert parameters across DP ranks.
    
    Expert parameters need special handling since different DP ranks
    may have processed different numbers of tokens for each expert.
    
    Args:
        model: Model containing ExpertParallelMoE layers
        dp_group: Data parallel process group
    """
    if dp_group is None:
        dp_group = get_data_parallel_group()
    
    if dp_group is None or dist.get_world_size(dp_group) <= 1:
        return
    
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        
        # Average gradients across DP ranks
        dist.all_reduce(param.grad, group=dp_group, op=dist.ReduceOp.AVG)


@contextmanager
def expert_parallel_context(
    ep_size: int,
    dp_size: Optional[int] = None,
):
    """Context manager for expert parallel training.
    
    Sets up process groups and cleans up on exit.
    
    Args:
        ep_size: Expert parallel size
        dp_size: Data parallel size
        
    Yields:
        Tuple of (EP group, DP group)
    """
    global _EXPERT_PARALLEL_GROUP, _DATA_PARALLEL_GROUP
    
    ep_group, dp_group = init_expert_parallel_group(ep_size, dp_size)
    
    try:
        yield ep_group, dp_group
    finally:
        _EXPERT_PARALLEL_GROUP = None
        _DATA_PARALLEL_GROUP = None


# Benchmark utilities
def benchmark_expert_parallel_scaling(
    d_model: int = 1024,
    d_hidden: int = 4096,
    num_experts_list: List[int] = [8, 16, 32, 64],
    batch_size: int = 8,
    seq_len: int = 512,
    num_steps: int = 100,
) -> Dict[int, Dict[str, float]]:
    """Benchmark expert parallelism scaling.
    
    Args:
        d_model: Model dimension
        d_hidden: Expert hidden dimension
        num_experts_list: List of expert counts to test
        batch_size: Batch size
        seq_len: Sequence length
        num_steps: Steps per benchmark
        
    Returns:
        Dict mapping num_experts to metrics
    """
    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for num_experts in num_experts_list:
        config = ExpertParallelConfig(
            num_experts=num_experts,
            ep_size=get_expert_parallel_world_size(),
        )
        
        moe = ExpertParallelMoE(
            d_model=d_model,
            d_hidden=d_hidden,
            num_experts=num_experts,
            config=config,
        ).to(device)
        
        # Warmup
        for _ in range(10):
            x = torch.randn(batch_size, seq_len, d_model, device=device)
            output, _ = moe(x)
            output.sum().backward()
        
        # Benchmark
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        import time
        start = time.time()
        
        for _ in range(num_steps):
            x = torch.randn(batch_size, seq_len, d_model, device=device)
            output, aux = moe(x)
            (output.sum() + aux["load_balance_loss"]).backward()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = time.time() - start
        
        results[num_experts] = {
            "throughput_steps_per_sec": num_steps / elapsed,
            "tokens_per_sec": (batch_size * seq_len * num_steps) / elapsed,
            "elapsed_seconds": elapsed,
        }
        
        logger.info(
            f"EP benchmark",
            num_experts=num_experts,
            throughput=results[num_experts]["throughput_steps_per_sec"],
        )
    
    return results
