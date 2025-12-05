"""
Multi-Latent Attention (MLA) for DeepSeek

This module implements DeepSeek's Multi-Latent Attention mechanism with:
- KV compression for memory efficiency
- Decoupled RoPE (separate positional encoding from content attention)
- Flash Attention integration via scaled_dot_product_attention
- KV Cache for efficient inference

MLA significantly reduces KV cache size while maintaining attention quality
through low-rank compression of key-value representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from dataclasses import dataclass
from deepseek.torch.model.parallel import ColumnParallelLinear, RowParallelLinear
from deepseek.torch.model.attention import (
    AttentionBackend,
    FlashAttentionConfig,
    scaled_dot_product_attention_with_backend,
    chunked_attention,
    get_optimal_attention_backend,
)

class KVCache:
    """
    Key-Value Cache for efficient generation.
    """
    def __init__(self, max_batch_size, max_seq_len, n_heads, head_dim, dtype=torch.float32, device='cpu'):
        self.k_cache = torch.zeros(max_batch_size, n_heads, max_seq_len, head_dim, dtype=dtype, device=device)
        self.v_cache = torch.zeros(max_batch_size, n_heads, max_seq_len, head_dim, dtype=dtype, device=device)
        self.current_seq_len = 0
        
    def update(self, k, v):
        """
        Update cache with new k, v.
        k, v: (B, H, S_new, D)
        """
        batch_size, n_heads, seq_len, head_dim = k.shape
        start_pos = self.current_seq_len
        end_pos = start_pos + seq_len
        
        self.k_cache[:batch_size, :, start_pos:end_pos, :] = k
        self.v_cache[:batch_size, :, start_pos:end_pos, :] = v
        
        self.current_seq_len = end_pos
        
        return self.k_cache[:batch_size, :, :end_pos, :], self.v_cache[:batch_size, :, :end_pos, :]


class LatentKVCache:
    """
    Latent KV Cache for MLA - stores compressed latent representations.
    
    Instead of storing full K/V tensors (batch, heads, seq_len, head_dim),
    this cache stores the compressed latent C_KV (batch, seq_len, d_latent),
    achieving approximately 14× memory reduction.
    
    The latent is up-projected to K/V on-demand during attention computation.
    """
    def __init__(
        self, 
        max_batch_size: int, 
        max_seq_len: int, 
        d_latent: int, 
        dtype=torch.float32, 
        device='cpu'
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.d_latent = d_latent
        self.dtype = dtype
        self.device = device
        
        # Store compressed latent instead of full K/V
        # Shape: (batch, max_seq_len, d_latent) instead of 2 × (batch, heads, seq_len, head_dim)
        self.latent_cache = torch.zeros(
            max_batch_size, max_seq_len, d_latent, 
            dtype=dtype, device=device
        )
        self.current_seq_len = 0
        
    def update(self, c_kv: torch.Tensor) -> torch.Tensor:
        """
        Update cache with new compressed latent.
        
        Args:
            c_kv: Compressed KV latent (batch, seq_len, d_latent)
            
        Returns:
            Full cached latent up to current position (batch, total_seq_len, d_latent)
        """
        batch_size, seq_len, d_latent = c_kv.shape
        start_pos = self.current_seq_len
        end_pos = start_pos + seq_len
        
        if end_pos > self.max_seq_len:
            raise ValueError(
                f"Sequence length {end_pos} exceeds max_seq_len {self.max_seq_len}"
            )
        
        self.latent_cache[:batch_size, start_pos:end_pos, :] = c_kv
        self.current_seq_len = end_pos
        
        return self.latent_cache[:batch_size, :end_pos, :]
    
    def get_cached_latent(self, batch_size: int) -> torch.Tensor:
        """Get the cached latent up to current position."""
        return self.latent_cache[:batch_size, :self.current_seq_len, :]
    
    def reset(self):
        """Reset cache for new generation."""
        self.current_seq_len = 0
        self.latent_cache.zero_()
    
    def memory_usage_bytes(self) -> int:
        """Return memory usage in bytes."""
        return self.latent_cache.element_size() * self.latent_cache.nelement()
    
    @staticmethod
    def compare_memory_savings(
        batch_size: int,
        seq_len: int, 
        n_heads: int,
        head_dim: int,
        d_latent: int,
        dtype=torch.float32
    ) -> dict:
        """
        Compare memory usage between standard KV cache and latent cache.
        
        Returns dict with memory comparison statistics.
        """
        element_size = torch.tensor([], dtype=dtype).element_size()
        
        # Standard KV cache: 2 × (batch, heads, seq, head_dim)
        standard_kv_bytes = 2 * batch_size * n_heads * seq_len * head_dim * element_size
        
        # Latent cache: (batch, seq, d_latent)
        latent_bytes = batch_size * seq_len * d_latent * element_size
        
        return {
            "standard_kv_bytes": standard_kv_bytes,
            "latent_cache_bytes": latent_bytes,
            "memory_reduction_ratio": standard_kv_bytes / latent_bytes if latent_bytes > 0 else float('inf'),
            "savings_bytes": standard_kv_bytes - latent_bytes,
            "savings_percent": (1 - latent_bytes / standard_kv_bytes) * 100 if standard_kv_bytes > 0 else 0
        }


# ============================================================================
# RoPE Implementations
# ============================================================================

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=2048):
        super().__init__()
        self.d_model = d_model
        # Create inverse frequency bands
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, seq_len=None):
        # x: (B, H, Seq, D)
        if seq_len is None:
            seq_len = x.shape[2]
            
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq) # (Seq, D/2)
        emb = torch.cat((freqs, freqs), dim=-1) # (Seq, D)
        
        # Reshape for broadcasting: (1, 1, Seq, D)
        return emb.unsqueeze(0).unsqueeze(0)


class ExtendedRoPEConfig:
    """Configuration for extended RoPE with NTK/YaRN scaling."""
    
    def __init__(
        self,
        d_head: int = 64,
        max_seq_len: int = 131072,
        base: float = 10000.0,
        scaling_type: str = "yarn",  # "none", "ntk", "yarn"
        ntk_alpha: float = 8.0,
        yarn_scale: float = 1.0,
        yarn_original_max_position: int = 4096,
        yarn_beta_fast: float = 32.0,
        yarn_beta_slow: float = 1.0,
    ):
        self.d_head = d_head
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_type = scaling_type
        self.ntk_alpha = ntk_alpha
        self.yarn_scale = yarn_scale
        self.yarn_original_max_position = yarn_original_max_position
        self.yarn_beta_fast = yarn_beta_fast
        self.yarn_beta_slow = yarn_beta_slow


class ExtendedRotaryPositionalEncoding(nn.Module):
    """
    Extended RoPE with NTK-aware scaling and YaRN interpolation.
    
    For 128K context support in DeepSeek-V3.2.
    
    Supports:
    - Standard RoPE
    - NTK-aware scaling (increases base frequency)
    - YaRN interpolation (blends frequencies for long context)
    """
    
    def __init__(self, config: ExtendedRoPEConfig):
        super().__init__()
        self.config = config
        self.dim = config.d_head
        
        # Compute base frequencies with scaling
        if config.scaling_type == "ntk":
            scaled_base = config.base * (config.ntk_alpha ** (self.dim / (self.dim - 2)))
            inv_freq = 1.0 / (scaled_base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        elif config.scaling_type == "yarn":
            inv_freq = self._compute_yarn_inv_freq()
        else:
            inv_freq = 1.0 / (config.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        
        self.register_buffer('inv_freq', inv_freq)
        
        # YaRN attention scaling
        if config.scaling_type == "yarn":
            self.yarn_attn_factor = 0.1 * math.log(
                config.max_seq_len / config.yarn_original_max_position
            ) + 1.0
        else:
            self.yarn_attn_factor = 1.0
        
        # Cache
        self._cos_cache: Optional[torch.Tensor] = None
        self._sin_cache: Optional[torch.Tensor] = None
        self._cache_seq_len = 0
    
    def _compute_yarn_inv_freq(self) -> torch.Tensor:
        """Compute inverse frequencies with YaRN interpolation."""
        config = self.config
        base_inv_freq = 1.0 / (config.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        scale = config.max_seq_len / config.yarn_original_max_position
        
        freqs = torch.arange(0, self.dim // 2).float()
        low = torch.clamp((freqs - config.yarn_beta_slow) / (config.yarn_beta_fast - config.yarn_beta_slow), 0.0, 1.0)
        
        inv_freq = base_inv_freq * (1 - low) + (base_inv_freq / scale) * low
        return inv_freq
    
    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update cos/sin cache."""
        if seq_len <= self._cache_seq_len and self._cos_cache is not None:
            return
        
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        self._cos_cache = emb.cos().to(dtype)
        self._sin_cache = emb.sin().to(dtype)
        self._cache_seq_len = seq_len
    
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Apply extended rotary positional encoding.
        
        Args:
            x: Input tensor [batch, heads, seq_len, dim]
            offset: Position offset for KV cache
            
        Returns:
            Position-encoded tensor
        """
        seq_len = x.shape[2]
        self._update_cache(offset + seq_len, x.device, x.dtype)
        
        cos = self._cos_cache[offset:offset + seq_len].unsqueeze(0).unsqueeze(0)
        sin = self._sin_cache[offset:offset + seq_len].unsqueeze(0).unsqueeze(0)
        
        x1, x2 = x.chunk(2, dim=-1)
        rotated = torch.cat([-x2, x1], dim=-1)
        
        result = x * cos + rotated * sin
        
        if self.config.scaling_type == "yarn":
            result = result * self.yarn_attn_factor
        
        return result

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, pos_emb):
    # x: (B, H, Seq, D)
    # pos_emb: (1, 1, Seq, D)
    return (x * pos_emb.cos()) + (rotate_half(x) * pos_emb.sin())

class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) with Flash Attention support.
    
    MLA compresses key-value representations through low-rank projection,
    significantly reducing KV cache size while maintaining attention quality.
    
    Features:
    - KV compression through down/up projection
    - Decoupled RoPE (separate positional encoding from content)
    - Flash Attention integration for efficient computation
    - KV cache support for inference
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_latent: int,
        d_rope: int,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        attention_config: Optional[FlashAttentionConfig] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_latent = d_latent
        self.d_rope = d_rope
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        
        # Attention configuration for Flash Attention
        self.attn_config = attention_config or FlashAttentionConfig()
        
        # Down-projection (Compression)
        self.kv_down_proj = ColumnParallelLinear(d_model, d_latent, bias=False)
        
        # Query projection
        self.q_proj = ColumnParallelLinear(d_model, d_model, bias=False)
        
        # KV Up-projections (Decompression)
        self.kv_up_proj_v = ColumnParallelLinear(d_latent, num_heads * self.head_dim, bias=False)
        self.kv_up_proj_k = ColumnParallelLinear(d_latent, num_heads * (self.head_dim - d_rope), bias=False)
        self.kv_up_proj_k_pe = ColumnParallelLinear(d_latent, d_rope, bias=False)
        
        # Output projection
        self.o_proj = RowParallelLinear(d_model, d_model, bias=False, input_is_parallel=True)
        
        # RoPE for positional encoding
        self.rope = RotaryPositionalEncoding(d_rope, max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # 1. Query Processing
        q = self.q_proj(x)  # (B, Seq, D)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, Seq, D_head)
        
        # Split Q into PE (positional) and Content parts (Decoupled RoPE)
        q_pe = q[..., :self.d_rope]
        q_unpe = q[..., self.d_rope:]
        
        # Apply RoPE to Q_pe
        if kv_cache:
            start_pos = kv_cache.current_seq_len
            pos_emb = self.rope(q_pe, seq_len + start_pos)
            pos_emb = pos_emb[:, :, start_pos:, :]
        else:
            pos_emb = self.rope(q_pe)
            
        q_pe = apply_rotary_pos_emb(q_pe, pos_emb)
        
        # Reassemble Q with decoupled RoPE
        q = torch.cat([q_pe, q_unpe], dim=-1)
        
        # 2. KV Processing (Compressed)
        latent = self.kv_down_proj(x)  # (B, Seq, D_latent)
        
        # Decompress V
        v = self.kv_up_proj_v(latent)  # (B, Seq, H * D_head)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Decompress K Content
        k_unpe = self.kv_up_proj_k(latent)  # (B, Seq, H * (D_head - D_rope))
        k_unpe = k_unpe.view(batch_size, seq_len, self.num_heads, self.head_dim - self.d_rope).transpose(1, 2)
        
        # Decompress K RoPE (Shared across heads)
        k_pe = self.kv_up_proj_k_pe(latent)  # (B, Seq, D_rope)
        k_pe = k_pe.unsqueeze(1)  # (B, 1, Seq, D_rope)
        k_pe = apply_rotary_pos_emb(k_pe, pos_emb)
        # Broadcast K_pe to all heads
        k_pe = k_pe.expand(-1, self.num_heads, -1, -1)
        
        # Concatenate K (positional + content)
        k = torch.cat([k_pe, k_unpe], dim=-1)
        
        # KV Cache Update (for inference)
        if kv_cache:
            k, v = kv_cache.update(k, v)
        
        # 3. Attention with Flash Attention support
        # Determine if we can use causal mask (only when no explicit mask and not using cache mid-sequence)
        use_causal = is_causal and mask is None and kv_cache is None
        
        # Dropout only during training
        dropout_p = self.dropout if self.training else 0.0
        
        # Use Flash Attention via scaled_dot_product_attention
        if self.attn_config.chunk_size and seq_len > self.attn_config.chunk_size:
            output = chunked_attention(
                q, k, v,
                chunk_size=self.attn_config.chunk_size,
                is_causal=use_causal,
                backend=self.attn_config.backend,
            )
        else:
            output = scaled_dot_product_attention_with_backend(
                q, k, v,
                attn_mask=mask,
                dropout_p=dropout_p,
                is_causal=use_causal,
                backend=self.attn_config.backend,
            )
        
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.o_proj(output)
    
    def forward_with_latent_cache(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        latent_cache: Optional[LatentKVCache] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass using LatentKVCache for memory-efficient inference.
        
        This method stores compressed latents (d_latent) instead of full K/V tensors,
        achieving approximately 14× memory reduction for KV cache.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask
            latent_cache: LatentKVCache for compressed KV storage
            is_causal: Whether to apply causal masking
            
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. Query Processing
        q = self.q_proj(x)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Split Q into PE (positional) and Content parts
        q_pe = q[..., :self.d_rope]
        q_unpe = q[..., self.d_rope:]
        
        # Determine position offset from cache
        if latent_cache is not None:
            start_pos = latent_cache.current_seq_len
            pos_emb = self.rope(q_pe, seq_len + start_pos)
            pos_emb = pos_emb[:, :, start_pos:, :]
        else:
            start_pos = 0
            pos_emb = self.rope(q_pe)
        
        # Apply RoPE to Q_pe
        q_pe = apply_rotary_pos_emb(q_pe, pos_emb)
        q = torch.cat([q_pe, q_unpe], dim=-1)
        
        # 2. KV Processing - Compress to latent
        c_kv = self.kv_down_proj(x)  # (B, Seq, D_latent)
        
        # Update latent cache if provided
        if latent_cache is not None:
            c_kv_full = latent_cache.update(c_kv)  # Returns full cached latent
        else:
            c_kv_full = c_kv
        
        full_seq_len = c_kv_full.shape[1]
        
        # Up-project cached latents to K and V on-demand
        # This is where the memory savings come from - we store d_latent but compute full K/V
        v = self.kv_up_proj_v(c_kv_full)
        v = v.view(batch_size, full_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        k_unpe = self.kv_up_proj_k(c_kv_full)
        k_unpe = k_unpe.view(batch_size, full_seq_len, self.num_heads, self.head_dim - self.d_rope).transpose(1, 2)
        
        # K positional encoding needs full sequence positions
        k_pe = self.kv_up_proj_k_pe(c_kv_full)
        k_pe = k_pe.unsqueeze(1)
        
        # Apply RoPE with full position range
        full_pos_emb = self.rope(k_pe, full_seq_len)
        k_pe = apply_rotary_pos_emb(k_pe, full_pos_emb)
        k_pe = k_pe.expand(-1, self.num_heads, -1, -1)
        
        k = torch.cat([k_pe, k_unpe], dim=-1)
        
        # 3. Attention computation
        use_causal = is_causal and mask is None and latent_cache is None
        dropout_p = self.dropout if self.training else 0.0
        
        if self.attn_config.chunk_size and seq_len > self.attn_config.chunk_size:
            output = chunked_attention(
                q, k, v,
                chunk_size=self.attn_config.chunk_size,
                is_causal=use_causal,
                backend=self.attn_config.backend,
            )
        else:
            output = scaled_dot_product_attention_with_backend(
                q, k, v,
                attn_mask=mask,
                dropout_p=dropout_p,
                is_causal=use_causal,
                backend=self.attn_config.backend,
            )
        
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.o_proj(output)

class DeepSeekAttention(nn.Module):
    """
    DeepSeek Attention wrapper for MLA with Flash Attention support.
    
    This provides a simple interface to the Multi-Head Latent Attention
    mechanism used in DeepSeek models.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_latent: int,
        d_rope: int,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        attention_config: Optional[FlashAttentionConfig] = None,
    ):
        super().__init__()
        self.mla = MultiHeadLatentAttention(
            d_model=d_model,
            num_heads=num_heads,
            d_latent=d_latent,
            d_rope=d_rope,
            max_seq_len=max_seq_len,
            dropout=dropout,
            attention_config=attention_config,
        )
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        return self.mla(x, mask, kv_cache, is_causal)
