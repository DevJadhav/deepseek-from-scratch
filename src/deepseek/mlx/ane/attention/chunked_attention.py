"""
ANE-Optimized Chunked Attention

This module provides chunked attention implementations optimized for ANE:
- Fixed 128-token windows for ANE tile size constraints
- Online softmax for memory-efficient attention
- Causal masking support

Key optimizations:
- Chunk size aligned to ANE tile size (128)
- FP16 computation for ANE efficiency
- Online softmax accumulation (FlashAttention-style)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def chunked_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int = 128,
    causal: bool = True,
    scale: float | None = None,
    use_fp16: bool = True,
) -> torch.Tensor:
    """
    Compute attention using chunked processing for ANE efficiency.

    This implements a memory-efficient attention pattern by processing
    the attention in fixed-size chunks, which:
    1. Fits within ANE tile size constraints
    2. Enables larger effective context without OOM
    3. Uses online softmax for numerical stability

    Args:
        q: Query tensor (batch, heads, seq_len, d_head)
        k: Key tensor (batch, heads, kv_len, d_head)
        v: Value tensor (batch, heads, kv_len, d_head)
        chunk_size: Size of attention chunks (default 128 for ANE)
        causal: Whether to apply causal masking
        scale: Attention scale factor (default: 1/sqrt(d_head))
        use_fp16: Use FP16 for computation

    Returns:
        Attention output (batch, heads, seq_len, d_head)
    """
    batch, heads, seq_len, d_head = q.shape
    kv_len = k.shape[2]

    if scale is None:
        scale = 1.0 / math.sqrt(d_head)

    # Convert to FP16 if needed
    original_dtype = q.dtype
    if use_fp16 and q.dtype != torch.float16:
        q = q.half()
        k = k.half()
        v = v.half()

    # Initialize output accumulator and normalizer
    output = torch.zeros_like(q)
    # For online softmax: track max and sum of exp for each query
    max_scores = torch.full(
        (batch, heads, seq_len, 1),
        float("-inf"),
        device=q.device,
        dtype=q.dtype,
    )
    sum_exp = torch.zeros(
        (batch, heads, seq_len, 1), device=q.device, dtype=q.dtype
    )

    # Process K/V in chunks
    for kv_start in range(0, kv_len, chunk_size):
        kv_end = min(kv_start + chunk_size, kv_len)
        k_chunk = k[:, :, kv_start:kv_end, :]
        v_chunk = v[:, :, kv_start:kv_end, :]

        # Compute attention scores for this chunk
        # (batch, heads, seq_len, chunk_size)
        scores = torch.matmul(q, k_chunk.transpose(-2, -1)) * scale

        # Apply causal mask if needed
        if causal:
            # Create causal mask for this chunk
            q_positions = torch.arange(seq_len, device=q.device)
            k_positions = torch.arange(kv_start, kv_end, device=q.device)
            # Mask where q_pos < k_pos (can't attend to future)
            causal_mask = q_positions.unsqueeze(1) < k_positions.unsqueeze(0)
            scores = scores.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

        # Online softmax update
        # Track running max and sum of exp(scores - max)
        chunk_max = scores.max(dim=-1, keepdim=True).values
        new_max = torch.maximum(max_scores, chunk_max)

        # Update exponential sums with correction factor
        exp_correction = torch.exp(max_scores - new_max)
        chunk_exp = torch.exp(scores - new_max)
        chunk_sum = chunk_exp.sum(dim=-1, keepdim=True)

        # Update running sum with correction
        sum_exp = sum_exp * exp_correction + chunk_sum
        max_scores = new_max

        # Compute weighted values for this chunk
        weights = chunk_exp / (sum_exp + 1e-8)
        output = output * exp_correction + torch.matmul(weights, v_chunk)

    # Normalize output
    output = output

    # Convert back to original dtype if needed
    if output.dtype != original_dtype:
        output = output.to(original_dtype)

    return output


class ANEChunkedAttention(nn.Module):
    """
    ANE-Optimized Chunked Attention Module.

    Implements chunked attention with:
    - 128-token chunks aligned to ANE tile size
    - Online softmax for memory efficiency
    - Optional causal masking
    - FP16 computation

    Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │ Chunked Attention Processing                                    │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  Q [seq_len × d] ──────────────────────────┐                   │
    │                                             │                   │
    │  K [kv_len × d] ──┬──► Chunk 0 ──┐         │                   │
    │                   ├──► Chunk 1 ──┼──► Attn ┼──► Output         │
    │                   └──► Chunk N ──┘         │                   │
    │                                             │                   │
    │  V [kv_len × d] ──┬──► Chunk 0 ──┐         │                   │
    │                   ├──► Chunk 1 ──┼─────────┘                   │
    │                   └──► Chunk N ──┘                              │
    │                                                                 │
    │  Online Softmax: max_i, sum_i updated per chunk                │
    └─────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        d_head: int,
        chunk_size: int = 128,
        causal: bool = True,
        use_fp16: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_head = d_head
        self.chunk_size = chunk_size
        self.causal = causal
        self.use_fp16 = use_fp16
        self.scale = 1.0 / math.sqrt(d_head)

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute chunked attention.

        Args:
            q: Query tensor (batch, heads, seq_len, d_head)
            k: Key tensor (batch, heads, kv_len, d_head)
            v: Value tensor (batch, heads, kv_len, d_head)
            mask: Optional attention mask

        Returns:
            Attention output (batch, heads, seq_len, d_head)
        """
        batch, heads, seq_len, d_head = q.shape
        kv_len = k.shape[2]

        # For short sequences, use standard attention
        if kv_len <= self.chunk_size:
            return self._standard_attention(q, k, v, mask)

        # For long sequences, use chunked attention
        return self._chunked_attention(q, k, v, mask)

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Standard attention for short sequences."""
        # Convert to FP16 if needed
        original_dtype = q.dtype
        if self.use_fp16 and q.dtype != torch.float16:
            q = q.half()
            k = k.half()
            v = v.half()

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask
        if self.causal:
            seq_len = q.shape[2]
            kv_len = k.shape[2]
            causal_mask = torch.triu(
                torch.ones(seq_len, kv_len, device=q.device, dtype=torch.bool),
                diagonal=kv_len - seq_len + 1,
            )
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Apply additional mask if provided
        if mask is not None:
            scores = scores + mask

        # Softmax and dropout
        weights = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            weights = self.dropout(weights)

        # Compute output
        output = torch.matmul(weights, v)

        # Convert back to original dtype
        if output.dtype != original_dtype:
            output = output.to(original_dtype)

        return output

    def _chunked_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Chunked attention for long sequences."""
        return chunked_attention_forward(
            q,
            k,
            v,
            chunk_size=self.chunk_size,
            causal=self.causal,
            scale=self.scale,
            use_fp16=self.use_fp16,
        )

    def extra_repr(self) -> str:
        return (
            f"d_head={self.d_head}, chunk_size={self.chunk_size}, "
            f"causal={self.causal}"
        )


def compute_attention_memory_savings(
    seq_len: int,
    chunk_size: int = 128,
    dtype_bytes: int = 2,  # FP16
) -> dict:
    """
    Compute memory savings from chunked attention.

    Standard attention requires O(seq_len^2) memory for attention matrix.
    Chunked attention requires O(seq_len × chunk_size) memory.

    Args:
        seq_len: Sequence length
        chunk_size: Chunk size
        dtype_bytes: Bytes per element (2 for FP16, 4 for FP32)

    Returns:
        Dict with memory comparison
    """
    standard_attn_memory = seq_len * seq_len * dtype_bytes
    chunked_attn_memory = seq_len * chunk_size * dtype_bytes

    return {
        "standard_memory_bytes": standard_attn_memory,
        "chunked_memory_bytes": chunked_attn_memory,
        "memory_reduction_ratio": standard_attn_memory / chunked_attn_memory,
        "savings_percent": (
            (1 - chunked_attn_memory / standard_attn_memory) * 100
        ),
    }
