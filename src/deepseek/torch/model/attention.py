"""
Attention Mechanisms for DeepSeek

This module implements various attention mechanisms with Flash Attention support:
- Multi-Query Attention (MQA)
- Grouped-Query Attention (GQA)
- Flash Attention integration via torch.nn.functional.scaled_dot_product_attention

Flash Attention is automatically enabled when available (CUDA SM 80+).
Fallback to memory-efficient or math attention on older hardware.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Literal
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# Flash Attention Configuration and Detection
# =============================================================================

class AttentionBackend(Enum):
    """Available attention backends."""
    FLASH = "flash"
    MEMORY_EFFICIENT = "memory_efficient"
    MATH = "math"
    AUTO = "auto"


@dataclass
class FlashAttentionConfig:
    """Configuration for Flash Attention."""
    backend: AttentionBackend = AttentionBackend.AUTO
    enable_flash: bool = True
    enable_mem_efficient: bool = True
    enable_math: bool = True
    # Chunk size for sequence length chunking (None = no chunking)
    chunk_size: Optional[int] = None
    # Dropout probability (0.0 during inference)
    dropout_p: float = 0.0
    # Whether attention is causal (autoregressive)
    is_causal: bool = True
    # Scale factor (None = 1/sqrt(head_dim))
    scale: Optional[float] = None


def detect_flash_attention_version() -> tuple[int, int] | None:
    """
    Detect Flash Attention version based on CUDA capability.
    
    Returns:
        Tuple of (major, minor) version or None if not available.
        - (2, 0): Flash Attention 2 (SM 80+, Ampere/Ada/Hopper)
        - (3, 0): Flash Attention 3 (SM 90+, Hopper)
        - None: Not available
    """
    if not torch.cuda.is_available():
        return None
    
    try:
        # Get compute capability of current device
        device = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device)
        
        # Flash Attention 3 requires Hopper (SM 90+)
        if major >= 9:
            return (3, 0)
        # Flash Attention 2 requires Ampere (SM 80+)
        elif major >= 8:
            return (2, 0)
        else:
            return None
    except Exception:
        return None


def get_optimal_attention_backend() -> AttentionBackend:
    """
    Determine the optimal attention backend for current hardware.
    
    Returns:
        The optimal AttentionBackend for the current device.
    """
    fa_version = detect_flash_attention_version()
    
    if fa_version is not None:
        return AttentionBackend.FLASH
    elif torch.cuda.is_available():
        # CUDA available but no Flash Attention - use memory efficient
        return AttentionBackend.MEMORY_EFFICIENT
    else:
        # CPU or MPS - use math backend
        return AttentionBackend.MATH


def scaled_dot_product_attention_with_backend(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    backend: AttentionBackend = AttentionBackend.AUTO,
) -> torch.Tensor:
    """
    Scaled dot-product attention with configurable backend.
    
    Wraps torch.nn.functional.scaled_dot_product_attention with explicit
    backend selection for Flash Attention, memory-efficient, or math kernels.
    
    Args:
        query: Query tensor [B, H, T, D]
        key: Key tensor [B, H, S, D]
        value: Value tensor [B, H, S, D]
        attn_mask: Optional attention mask
        dropout_p: Dropout probability
        is_causal: Whether to use causal masking
        scale: Optional scale factor (default: 1/sqrt(D))
        backend: Which attention backend to use
        
    Returns:
        Attention output tensor [B, H, T, D]
    """
    if backend == AttentionBackend.AUTO:
        backend = get_optimal_attention_backend()
    
    # For non-CUDA devices, always use math backend
    if not query.is_cuda:
        backend = AttentionBackend.MATH
    
    # Use torch's sdp_kernel context manager to select backend
    if backend == AttentionBackend.FLASH:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=False,
            enable_mem_efficient=False,
        ):
            return F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
            )
    elif backend == AttentionBackend.MEMORY_EFFICIENT:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_math=False,
            enable_mem_efficient=True,
        ):
            return F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
            )
    else:  # MATH backend
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=False,
        ):
            return F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
            )


def chunked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    chunk_size: int,
    is_causal: bool = True,
    backend: AttentionBackend = AttentionBackend.AUTO,
) -> torch.Tensor:
    """
    Chunked attention for very long sequences.
    
    Splits the sequence into chunks and processes each chunk separately
    to handle sequences that exceed Flash Attention's optimal range.
    
    Args:
        query: Query tensor [B, H, T, D]
        key: Key tensor [B, H, S, D]
        value: Value tensor [B, H, S, D]
        chunk_size: Size of each chunk
        is_causal: Whether attention is causal
        backend: Attention backend to use
        
    Returns:
        Attention output tensor [B, H, T, D]
    """
    batch_size, num_heads, seq_len, head_dim = query.shape
    
    if seq_len <= chunk_size:
        return scaled_dot_product_attention_with_backend(
            query, key, value,
            is_causal=is_causal,
            backend=backend,
        )
    
    outputs = []
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, seq_len)
        
        q_chunk = query[:, :, start_idx:end_idx, :]
        
        if is_causal:
            # For causal attention, each chunk only attends to current and previous tokens
            k_chunk = key[:, :, :end_idx, :]
            v_chunk = value[:, :, :end_idx, :]
        else:
            # For non-causal, attend to all tokens
            k_chunk = key
            v_chunk = value
        
        chunk_output = scaled_dot_product_attention_with_backend(
            q_chunk, k_chunk, v_chunk,
            is_causal=is_causal and (i == 0),  # Only first chunk needs causal mask
            backend=backend,
        )
        outputs.append(chunk_output)
    
    return torch.cat(outputs, dim=2)

# =============================================================================
# Multi-Query Attention with Flash Attention Support
# =============================================================================

class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention (MQA) with Flash Attention support.
    
    Uses a single KV head shared across all query heads, significantly
    reducing KV cache size and memory bandwidth requirements.
    
    Flash Attention is automatically enabled when available (CUDA SM 80+).
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        attention_config: Optional[FlashAttentionConfig] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        
        # Attention configuration
        self.attn_config = attention_config or FlashAttentionConfig()
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.head_dim, bias=False)  # Single KV head
        self.v_proj = nn.Linear(d_model, self.head_dim, bias=False)  # Single KV head
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape Q: (B, Seq, H, D_head) -> (B, H, Seq, D_head)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Reshape K, V: (B, Seq, D_head) -> (B, 1, Seq, D_head) 
        # Then expand to (B, H, Seq, D_head) for Flash Attention compatibility
        k = k.view(batch_size, seq_len, 1, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, 1, self.head_dim).transpose(1, 2)
        
        # Expand K, V to match number of heads (Flash Attention requires same num heads)
        k = k.expand(-1, self.num_heads, -1, -1)
        v = v.expand(-1, self.num_heads, -1, -1)
        
        # Determine dropout (0 during eval)
        dropout_p = self.dropout if self.training else 0.0
        
        # Use Flash Attention via scaled_dot_product_attention
        if self.attn_config.chunk_size and seq_len > self.attn_config.chunk_size:
            output = chunked_attention(
                q, k, v,
                chunk_size=self.attn_config.chunk_size,
                is_causal=is_causal,
                backend=self.attn_config.backend,
            )
        else:
            output = scaled_dot_product_attention_with_backend(
                q, k, v,
                attn_mask=mask,
                dropout_p=dropout_p,
                is_causal=is_causal and mask is None,  # Don't double-apply causal mask
                backend=self.attn_config.backend,
            )
        
        # Reshape back: (B, H, Seq, D_head) -> (B, Seq, D)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.o_proj(output)


# =============================================================================
# Grouped-Query Attention with Flash Attention Support
# =============================================================================

class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) with Flash Attention support.
    
    Groups query heads to share KV heads, providing a balance between
    MHA (full KV per head) and MQA (single KV for all heads).
    
    Flash Attention is automatically enabled when available (CUDA SM 80+).
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_groups: int,
        dropout: float = 0.0,
        attention_config: Optional[FlashAttentionConfig] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = d_model // num_heads
        self.num_kv_heads = num_groups  # One KV head per group
        self.dropout = dropout
        
        # Attention configuration
        self.attn_config = attention_config or FlashAttentionConfig()
        
        # Check if heads divide evenly by groups
        assert num_heads % num_groups == 0, f"num_heads ({num_heads}) must be divisible by num_groups ({num_groups})"
        self.heads_per_group = num_heads // num_groups
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape Q: (B, Seq, H, D_head) -> (B, H, Seq, D_head)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Reshape K, V: (B, Seq, G, D_head) -> (B, G, Seq, D_head)
        k = k.view(batch_size, seq_len, self.num_groups, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_groups, self.head_dim).transpose(1, 2)
        
        # Expand K, V to match number of heads for Flash Attention
        # (B, G, Seq, D_head) -> (B, G, 1, Seq, D_head) -> (B, G, H_per_G, Seq, D_head) -> (B, H, Seq, D_head)
        k = k.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1)
        k = k.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
        v = v.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1)
        v = v.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
        
        # Determine dropout (0 during eval)
        dropout_p = self.dropout if self.training else 0.0
        
        # Use Flash Attention via scaled_dot_product_attention
        if self.attn_config.chunk_size and seq_len > self.attn_config.chunk_size:
            output = chunked_attention(
                q, k, v,
                chunk_size=self.attn_config.chunk_size,
                is_causal=is_causal,
                backend=self.attn_config.backend,
            )
        else:
            output = scaled_dot_product_attention_with_backend(
                q, k, v,
                attn_mask=mask,
                dropout_p=dropout_p,
                is_causal=is_causal and mask is None,  # Don't double-apply causal mask
                backend=self.attn_config.backend,
            )
        
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.o_proj(output)
