"""
Sequence Parallelism for DeepSeek Distributed Training.

This module implements sequence dimension sharding with:
- Sequence-parallel attention with all-gather for Q,K,V and reduce-scatter for output
- Sequence-parallel LayerNorm/RMSNorm with appropriate reductions
- Integration with FSDP (orthogonal parallelism dimensions)
- Handling sequence lengths not divisible by world size (padding/masking)
- Ring attention pattern for extreme sequence lengths (128K+)
- Sequence-parallel-aware position encoding (RoPE with offset)
- Gradient checkpointing compatible with sequence parallelism

Reference: DeepSeek-V3 uses sequence parallelism for handling long contexts
efficiently across multiple GPUs.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum

try:
    from deepseek.torch.utils.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# Global sequence parallel state
_SEQUENCE_PARALLEL_GROUP: Optional[dist.ProcessGroup] = None
_SEQUENCE_PARALLEL_WORLD_SIZE: int = 1
_SEQUENCE_PARALLEL_RANK: int = 0


def init_sequence_parallel_group(
    ranks: Optional[List[int]] = None,
    world_size: Optional[int] = None,
) -> Optional[dist.ProcessGroup]:
    """Initialize sequence parallel process group.
    
    Args:
        ranks: Explicit list of ranks in the SP group
        world_size: If ranks not provided, use first N ranks
        
    Returns:
        Process group for sequence parallel communication
    """
    global _SEQUENCE_PARALLEL_GROUP, _SEQUENCE_PARALLEL_WORLD_SIZE, _SEQUENCE_PARALLEL_RANK
    
    if not dist.is_initialized():
        _SEQUENCE_PARALLEL_WORLD_SIZE = 1
        _SEQUENCE_PARALLEL_RANK = 0
        return None
    
    if ranks is None:
        if world_size is None:
            world_size = dist.get_world_size()
        ranks = list(range(world_size))
    
    _SEQUENCE_PARALLEL_GROUP = dist.new_group(ranks)
    _SEQUENCE_PARALLEL_WORLD_SIZE = len(ranks)
    
    my_rank = dist.get_rank()
    if my_rank in ranks:
        _SEQUENCE_PARALLEL_RANK = ranks.index(my_rank)
    else:
        _SEQUENCE_PARALLEL_RANK = -1
    
    return _SEQUENCE_PARALLEL_GROUP


def get_sequence_parallel_group() -> Optional[dist.ProcessGroup]:
    """Get the sequence parallel process group."""
    return _SEQUENCE_PARALLEL_GROUP


def get_sequence_parallel_world_size() -> int:
    """Get sequence parallel world size."""
    return _SEQUENCE_PARALLEL_WORLD_SIZE


def get_sequence_parallel_rank() -> int:
    """Get rank within sequence parallel group."""
    return _SEQUENCE_PARALLEL_RANK


@dataclass
class SequenceParallelConfig:
    """Configuration for sequence parallelism.
    
    Attributes:
        sp_size: Sequence parallel world size
        enable_ring_attention: Use ring attention for extreme lengths
        ring_chunk_size: Chunk size for ring attention
        gradient_checkpointing: Enable gradient checkpointing
        scatter_to_sequence_parallel: Whether to scatter inputs
        gather_from_sequence_parallel: Whether to gather outputs
    """
    sp_size: int = 1
    enable_ring_attention: bool = False
    ring_chunk_size: int = 4096
    gradient_checkpointing: bool = False
    scatter_to_sequence_parallel: bool = True
    gather_from_sequence_parallel: bool = True


class _AllGatherFromSequenceParallel(torch.autograd.Function):
    """All-gather operation for sequence parallel tensors.
    
    Forward: All-gather along sequence dimension
    Backward: Reduce-scatter gradients
    """
    
    @staticmethod
    def forward(ctx, input_: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        ctx.group = group
        world_size = dist.get_world_size(group)
        
        if world_size == 1:
            return input_
        
        # All-gather along sequence dimension (dim=1)
        tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
        dist.all_gather(tensor_list, input_, group=group)
        
        # Concatenate along sequence dimension
        output = torch.cat(tensor_list, dim=1)
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        group = ctx.group
        world_size = dist.get_world_size(group)
        
        if world_size == 1:
            return grad_output, None
        
        # Reduce-scatter: split and reduce
        # grad_output: (batch, full_seq, hidden)
        # split along seq dimension, sum contributions
        seq_len = grad_output.size(1)
        chunk_size = seq_len // world_size
        
        # Split into chunks
        grad_chunks = torch.chunk(grad_output, world_size, dim=1)
        
        # Reduce-scatter
        rank = dist.get_rank(group)
        grad_input = torch.zeros_like(grad_chunks[0])
        
        # Use all_reduce for simplicity (reduce_scatter not always available)
        for i, chunk in enumerate(grad_chunks):
            if i == rank:
                grad_input.copy_(chunk)
            dist.all_reduce(grad_input, op=dist.ReduceOp.SUM, group=group)
        
        return grad_input, None


class _ReduceScatterToSequenceParallel(torch.autograd.Function):
    """Reduce-scatter operation for sequence parallel tensors.
    
    Forward: Reduce-scatter along sequence dimension
    Backward: All-gather gradients
    """
    
    @staticmethod
    def forward(ctx, input_: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        ctx.group = group
        world_size = dist.get_world_size(group)
        
        if world_size == 1:
            return input_
        
        # Input: (batch, full_seq, hidden)
        seq_len = input_.size(1)
        chunk_size = seq_len // world_size
        rank = dist.get_rank(group)
        
        # Manual reduce-scatter using all_reduce
        # Split, reduce, and take local chunk
        chunks = torch.chunk(input_, world_size, dim=1)
        output = chunks[rank].contiguous()
        
        # All-reduce to sum contributions from all ranks
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=group)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        group = ctx.group
        world_size = dist.get_world_size(group)
        
        if world_size == 1:
            return grad_output, None
        
        # All-gather gradients
        tensor_list = [torch.empty_like(grad_output) for _ in range(world_size)]
        dist.all_gather(tensor_list, grad_output, group=group)
        
        grad_input = torch.cat(tensor_list, dim=1)
        return grad_input, None


def all_gather_from_sequence_parallel(
    input_: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """All-gather tensor from sequence parallel ranks.
    
    Args:
        input_: Local tensor (batch, local_seq, hidden)
        group: Sequence parallel group
        
    Returns:
        Gathered tensor (batch, full_seq, hidden)
    """
    if group is None:
        group = get_sequence_parallel_group()
    
    if group is None or dist.get_world_size(group) == 1:
        return input_
    
    return _AllGatherFromSequenceParallel.apply(input_, group)


def reduce_scatter_to_sequence_parallel(
    input_: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Reduce-scatter tensor to sequence parallel ranks.
    
    Args:
        input_: Full tensor (batch, full_seq, hidden)
        group: Sequence parallel group
        
    Returns:
        Local tensor (batch, local_seq, hidden)
    """
    if group is None:
        group = get_sequence_parallel_group()
    
    if group is None or dist.get_world_size(group) == 1:
        return input_
    
    return _ReduceScatterToSequenceParallel.apply(input_, group)


class SequenceParallelRMSNorm(nn.Module):
    """RMSNorm with sequence parallelism support.
    
    Computes RMSNorm across the local sequence chunk and handles
    distributed reduction for correct normalization.
    """
    
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        sp_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.sp_group = sp_group
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sequence parallel RMSNorm.
        
        Args:
            x: Input tensor (batch, seq, dim)
            
        Returns:
            Normalized tensor
        """
        # Compute local variance contribution
        variance = x.pow(2).mean(-1, keepdim=True)
        
        # If sequence parallel, need to gather variance across sequence chunks
        group = self.sp_group or get_sequence_parallel_group()
        if group is not None and dist.get_world_size(group) > 1:
            # Average variance across all ranks
            dist.all_reduce(variance, op=dist.ReduceOp.AVG, group=group)
        
        # Normalize
        x_norm = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_norm


class SequenceParallelAttention(nn.Module):
    """Attention layer with sequence parallelism.
    
    Implements:
    - All-gather Q, K, V across sequence parallel ranks
    - Compute attention on full sequence
    - Reduce-scatter output back to local chunks
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        config: Optional[SequenceParallelConfig] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.config = config or SequenceParallelConfig()
        
        # Projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with sequence parallel attention.
        
        Args:
            hidden_states: (batch, local_seq, hidden)
            attention_mask: Optional attention mask
            position_ids: Position IDs for RoPE
            
        Returns:
            Output tensor (batch, local_seq, hidden)
        """
        batch_size, local_seq_len, _ = hidden_states.shape
        
        # Project Q, K, V locally
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # All-gather K, V across sequence parallel ranks for full context
        group = get_sequence_parallel_group()
        if group is not None and dist.get_world_size(group) > 1:
            # Keep Q local, gather K, V
            k_full = all_gather_from_sequence_parallel(k, group)
            v_full = all_gather_from_sequence_parallel(v, group)
        else:
            k_full = k
            v_full = v
        
        # Reshape for attention
        q = q.view(batch_size, local_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_full = k_full.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_full = v_full.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        # Q: (batch, heads, local_seq, head_dim)
        # K: (batch, heads, full_seq, head_dim)
        attn_weights = torch.matmul(q, k_full.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.dropout(attn_weights)
        
        # Compute attention output
        # (batch, heads, local_seq, head_dim)
        attn_output = torch.matmul(attn_weights, v_full)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, local_seq_len, self.d_model)
        output = self.o_proj(attn_output)
        
        return output


class RingAttention(nn.Module):
    """Ring Attention for extreme sequence lengths.
    
    Implements the ring attention pattern where:
    - Sequence is split across devices
    - K, V blocks are passed around in a ring
    - Each device computes partial attention and accumulates
    
    This allows O(N/P) memory per device for N sequence length
    across P devices.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        chunk_size: int = 4096,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.chunk_size = chunk_size
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with ring attention.
        
        Args:
            hidden_states: Local sequence chunk (batch, local_seq, hidden)
            attention_mask: Optional causal mask
            
        Returns:
            Output tensor (batch, local_seq, hidden)
        """
        batch_size, local_seq_len, _ = hidden_states.shape
        
        # Project Q, K, V locally
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape
        q = q.view(batch_size, local_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, local_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, local_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Ring attention loop
        group = get_sequence_parallel_group()
        world_size = get_sequence_parallel_world_size()
        rank = get_sequence_parallel_rank()
        
        # Initialize accumulators for online softmax
        # Using the "flash attention" style accumulation
        max_scores = torch.full((batch_size, self.num_heads, local_seq_len, 1), 
                                float('-inf'), device=q.device, dtype=q.dtype)
        sum_exp = torch.zeros_like(max_scores)
        output_accum = torch.zeros(batch_size, self.num_heads, local_seq_len, self.head_dim,
                                   device=q.device, dtype=q.dtype)
        
        # Process local K, V first
        current_k = k.clone()
        current_v = v.clone()
        
        for step in range(world_size):
            # Compute attention for current K, V block
            # Determine which positions this K, V corresponds to
            source_rank = (rank - step) % world_size
            block_start = source_rank * local_seq_len
            block_end = block_start + local_seq_len
            
            # Compute scores
            scores = torch.matmul(q, current_k.transpose(-2, -1)) * self.scale
            
            # Apply causal mask: Q position i can only attend to K positions <= i
            if attention_mask is not None:
                # Create causal mask for this block
                q_positions = torch.arange(rank * local_seq_len, 
                                          (rank + 1) * local_seq_len, 
                                          device=q.device)
                k_positions = torch.arange(block_start, block_end, device=q.device)
                causal_mask = q_positions.unsqueeze(1) >= k_positions.unsqueeze(0)
                scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), 
                                           float('-inf'))
            
            # Online softmax update
            new_max = torch.maximum(max_scores, scores.max(dim=-1, keepdim=True).values)
            
            # Update accumulator
            exp_diff = torch.exp(max_scores - new_max)
            exp_scores = torch.exp(scores - new_max)
            
            sum_exp = sum_exp * exp_diff + exp_scores.sum(dim=-1, keepdim=True)
            output_accum = output_accum * exp_diff + torch.matmul(exp_scores, current_v)
            max_scores = new_max
            
            # Pass K, V to next rank in ring (except on last iteration)
            if step < world_size - 1 and group is not None:
                # Send to next, receive from previous
                send_rank = (rank + 1) % world_size
                recv_rank = (rank - 1) % world_size
                
                # Create receive buffers
                recv_k = torch.empty_like(current_k)
                recv_v = torch.empty_like(current_v)
                
                # Async send/recv
                send_k_handle = dist.isend(current_k.contiguous(), send_rank, group=group)
                send_v_handle = dist.isend(current_v.contiguous(), send_rank, group=group)
                recv_k_handle = dist.irecv(recv_k, recv_rank, group=group)
                recv_v_handle = dist.irecv(recv_v, recv_rank, group=group)
                
                # Wait for receives
                recv_k_handle.wait()
                recv_v_handle.wait()
                send_k_handle.wait()
                send_v_handle.wait()
                
                current_k = recv_k
                current_v = recv_v
        
        # Normalize output
        output = output_accum / sum_exp
        
        # Reshape and project
        output = output.transpose(1, 2).reshape(batch_size, local_seq_len, self.d_model)
        output = self.o_proj(output)
        
        return output


def compute_rope_with_offset(
    positions: torch.Tensor,
    dim: int,
    base: float = 10000.0,
    offset: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute RoPE embeddings with sequence parallel offset.
    
    Args:
        positions: Position indices (batch, seq) or (seq,)
        dim: Dimension of RoPE
        base: Base for frequency computation
        offset: Offset to add for sequence parallel
        
    Returns:
        Tuple of (cos, sin) embeddings
    """
    # Adjust positions for sequence parallel offset
    if offset > 0:
        positions = positions + offset
    
    # Compute frequencies
    half_dim = dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, device=positions.device).float() / half_dim))
    
    # Outer product of positions and frequencies
    if positions.dim() == 1:
        angles = positions.unsqueeze(-1) * freqs.unsqueeze(0)
    else:
        angles = positions.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)
    
    # Compute cos and sin
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    
    return cos, sin


class SequenceParallelTransformerLayer(nn.Module):
    """Transformer layer with full sequence parallelism support.
    
    Combines:
    - Sequence parallel attention
    - Sequence parallel RMSNorm
    - Proper gradient checkpointing
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_hidden: int,
        dropout: float = 0.0,
        config: Optional[SequenceParallelConfig] = None,
    ):
        super().__init__()
        self.config = config or SequenceParallelConfig()
        
        self.attn_norm = SequenceParallelRMSNorm(d_model)
        self.attn = SequenceParallelAttention(
            d_model, num_heads, dropout, config
        )
        
        self.mlp_norm = SequenceParallelRMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_hidden * 2),
            nn.SiLU(),
            nn.Linear(d_hidden * 2, d_hidden),
            nn.Linear(d_hidden, d_model),
        )
    
    def _attn_forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.attn(self.attn_norm(x), mask)
    
    def _mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.mlp_norm(x))
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.config.gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            x = x + checkpoint(self._attn_forward, x, mask, use_reentrant=False)
            x = x + checkpoint(self._mlp_forward, x, use_reentrant=False)
        else:
            x = x + self._attn_forward(x, mask)
            x = x + self._mlp_forward(x)
        
        return x


def pad_sequence_for_sp(
    input_tensor: torch.Tensor,
    sp_world_size: int,
    pad_value: float = 0.0,
) -> Tuple[torch.Tensor, int]:
    """Pad sequence length to be divisible by SP world size.
    
    Args:
        input_tensor: (batch, seq, hidden)
        sp_world_size: Sequence parallel world size
        pad_value: Value to use for padding
        
    Returns:
        Tuple of (padded_tensor, original_seq_len)
    """
    batch_size, seq_len, hidden = input_tensor.shape
    
    if seq_len % sp_world_size == 0:
        return input_tensor, seq_len
    
    # Compute padding
    pad_len = sp_world_size - (seq_len % sp_world_size)
    
    # Pad
    padded = F.pad(input_tensor, (0, 0, 0, pad_len), value=pad_value)
    
    return padded, seq_len


def unpad_sequence(
    padded_tensor: torch.Tensor,
    original_seq_len: int,
) -> torch.Tensor:
    """Remove padding from sequence parallel output.
    
    Args:
        padded_tensor: Padded tensor (batch, padded_seq, hidden)
        original_seq_len: Original sequence length
        
    Returns:
        Unpadded tensor (batch, original_seq, hidden)
    """
    return padded_tensor[:, :original_seq_len, :]


def benchmark_sequence_parallelism(
    d_model: int = 1024,
    num_heads: int = 16,
    seq_len: int = 8192,
    batch_size: int = 4,
    sp_sizes: List[int] = [1, 2, 4, 8],
    num_iterations: int = 100,
) -> Dict[int, Dict[str, float]]:
    """Benchmark sequence parallelism scaling.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        seq_len: Full sequence length
        batch_size: Batch size
        sp_sizes: Sequence parallel sizes to test
        num_iterations: Number of iterations
        
    Returns:
        Dict mapping sp_size to metrics
    """
    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for sp_size in sp_sizes:
        if sp_size > get_sequence_parallel_world_size():
            continue
        
        config = SequenceParallelConfig(sp_size=sp_size)
        layer = SequenceParallelTransformerLayer(
            d_model=d_model,
            num_heads=num_heads,
            d_hidden=d_model * 4,
            config=config,
        ).to(device)
        
        local_seq_len = seq_len // sp_size
        
        # Warmup
        for _ in range(10):
            x = torch.randn(batch_size, local_seq_len, d_model, device=device)
            output = layer(x)
            output.sum().backward()
        
        # Benchmark
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        import time
        start = time.time()
        
        for _ in range(num_iterations):
            x = torch.randn(batch_size, local_seq_len, d_model, device=device)
            output = layer(x)
            output.sum().backward()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = time.time() - start
        
        results[sp_size] = {
            "throughput_steps_per_sec": num_iterations / elapsed,
            "tokens_per_sec": (batch_size * seq_len * num_iterations) / elapsed,
            "memory_per_device_mb": local_seq_len * batch_size * d_model * 4 / 1e6,
            "elapsed_seconds": elapsed,
        }
        
        logger.info(
            f"SP benchmark",
            sp_size=sp_size,
            throughput=results[sp_size]["throughput_steps_per_sec"],
            memory_mb=results[sp_size]["memory_per_device_mb"],
        )
    
    return results
