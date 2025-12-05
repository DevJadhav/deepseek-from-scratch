"""
ANE-Optimized Multi-Latent Attention (MLA)

This module implements DeepSeek's Multi-Latent Attention for Apple Neural Engine:
- KV compression for 14x memory reduction
- Decoupled RoPE for content and positional attention
- Chunked attention for ANE constraints
- Zero-copy KV cache

Key optimizations:
- FP16 computation throughout
- INT8 weight quantization support
- 128-token chunk size for ANE tiles
- Pre-computed RoPE frequencies
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.base import ANELinear
from .chunked_attention import ANEChunkedAttention
from .kv_cache import ANELatentKVCache
from .rope import ANERoPE, ANERoPEConfig, RoPEScalingType


@dataclass
class ANEMLAConfig:
    """Configuration for ANE-optimized Multi-Latent Attention."""

    d_model: int = 4096
    num_heads: int = 32
    d_latent: int = 512  # Latent dimension for KV compression
    d_rope: int = 64  # Dimension for decoupled RoPE
    d_head: int | None = None  # Head dimension (default: d_model // num_heads)
    chunk_size: int = 128  # ANE-friendly chunk size
    max_seq_len: int = 8192
    # ANE optimization
    use_fp16: bool = True
    use_quantized_weights: bool = False
    # RoPE scaling
    rope_scaling_type: RoPEScalingType = RoPEScalingType.NTK_AWARE
    rope_base: float = 10000.0
    # Dropout
    dropout: float = 0.0

    def __post_init__(self):
        if self.d_head is None:
            self.d_head = self.d_model // self.num_heads

    @classmethod
    def for_deepseek_v3(
        cls,
        d_model: int = 4096,
        num_heads: int = 32,
        max_seq_len: int = 131072,
    ) -> "ANEMLAConfig":
        """Create config matching DeepSeek-V3 architecture."""
        return cls(
            d_model=d_model,
            num_heads=num_heads,
            d_latent=d_model // 8,  # 8x compression
            d_rope=64,
            chunk_size=128,
            max_seq_len=max_seq_len,
            rope_scaling_type=RoPEScalingType.NTK_AWARE,
        )


class ANEMultiLatentAttention(nn.Module):
    """
    ANE-Optimized Multi-Latent Attention (MLA).

    MLA reduces KV cache memory by compressing key-value representations
    into a low-rank latent space while maintaining attention quality through
    decoupled positional encoding.

    Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │ Multi-Latent Attention                                          │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  Input x ──┬──► Q_proj ──► Split ──┬──► Q_pe (RoPE) ──┐        │
    │            │              heads   └──► Q_content ────┤        │
    │            │                                          │        │
    │            └──► KV_down ──► C_KV ──┬──► K_up ──► RoPE┼─► Attn │
    │                          (latent) └──► V_up ────────┤        │
    │                                                      │        │
    │  Memory Savings:                                     ▼        │
    │  • Standard: 2 × heads × d_head = 8192 per token    Out       │
    │  • Latent: d_latent = 512 per token                           │
    │  • Reduction: ~16x                                             │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

    Features:
    - Latent KV compression (14-16x memory reduction)
    - Decoupled RoPE (separate positional from content)
    - Chunked attention (128-token windows)
    - FP16 computation for ANE
    - Optional INT8 weight quantization
    """

    def __init__(self, config: ANEMLAConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.d_latent = config.d_latent
        self.d_rope = config.d_rope
        self.d_head = config.d_head or config.d_model // config.num_heads
        self.use_fp16 = config.use_fp16

        # Total head dimension = content + RoPE
        self.d_head_total = self.d_head + self.d_rope

        # Query projection: x -> Q (full dimension)
        self._init_query_projections(config)

        # KV compression: x -> C_KV (latent)
        self._init_kv_projections(config)

        # Output projection
        self._init_output_projection(config)

        # RoPE for positional encoding
        self._init_rope(config)

        # Chunked attention
        self.attention = ANEChunkedAttention(
            d_head=self.d_head_total,
            chunk_size=config.chunk_size,
            causal=True,
            use_fp16=config.use_fp16,
            dropout=config.dropout,
        )

        # KV cache (initialized on first forward or explicitly)
        self.kv_cache: ANELatentKVCache | None = None

        # Convert entire module to FP16 if requested
        if config.use_fp16:
            self.half()

    def _init_query_projections(self, config: ANEMLAConfig):
        """Initialize query projections."""
        LinearClass = ANELinear if config.use_quantized_weights else nn.Linear

        # Query content projection: d_model -> num_heads * d_head
        self.q_content_proj = LinearClass(
            config.d_model,
            config.num_heads * self.d_head,
            bias=False,
        )

        # Query RoPE projection: d_model -> num_heads * d_rope
        self.q_rope_proj = LinearClass(
            config.d_model,
            config.num_heads * config.d_rope,
            bias=False,
        )

    def _init_kv_projections(self, config: ANEMLAConfig):
        """Initialize KV compression projections."""
        LinearClass = ANELinear if config.use_quantized_weights else nn.Linear

        # Down projection: d_model -> d_latent
        self.kv_down_proj = LinearClass(
            config.d_model,
            config.d_latent,
            bias=False,
        )

        # Up projections from latent to full KV
        # K content: d_latent -> num_heads * d_head
        self.k_content_up = LinearClass(
            config.d_latent,
            config.num_heads * self.d_head,
            bias=False,
        )

        # K RoPE: d_latent -> d_rope (shared across heads)
        self.k_rope_up = LinearClass(
            config.d_latent,
            config.d_rope,
            bias=False,
        )

        # V: d_latent -> num_heads * d_head
        self.v_up = LinearClass(
            config.d_latent,
            config.num_heads * self.d_head,
            bias=False,
        )

    def _init_output_projection(self, config: ANEMLAConfig):
        """Initialize output projection."""
        LinearClass = ANELinear if config.use_quantized_weights else nn.Linear

        self.out_proj = LinearClass(
            config.num_heads * self.d_head,
            config.d_model,
            bias=False,
        )

    def _init_rope(self, config: ANEMLAConfig):
        """Initialize RoPE for positional encoding."""
        rope_config = ANERoPEConfig(
            d_head=config.d_rope,
            max_seq_len=config.max_seq_len,
            base=config.rope_base,
            scaling_type=config.rope_scaling_type,
            use_fp16=config.use_fp16,
        )
        self.rope = ANERoPE(rope_config)

    def init_kv_cache(
        self,
        batch_size: int,
        max_seq_len: int | None = None,
    ) -> ANELatentKVCache:
        """Initialize or reset the KV cache."""
        max_seq_len = max_seq_len or self.config.max_seq_len
        self.kv_cache = ANELatentKVCache(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            d_latent=self.d_latent,
            use_fp16=self.use_fp16,
        )
        return self.kv_cache

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        use_cache: bool = False,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass of Multi-Latent Attention.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask
            use_cache: Whether to use KV cache for inference
            position_offset: Position offset for RoPE (for KV cache)

        Returns:
            Tuple of (output, cached_latent) where cached_latent is
            the compressed KV representation if use_cache=True
        """
        batch, seq_len, d_model = x.shape

        # Convert to FP16 if needed
        original_dtype = x.dtype
        if self.use_fp16 and x.dtype != torch.float16:
            x = x.half()

        # Compute queries
        q_content, q_rope = self._compute_queries(x, position_offset)

        # Compute KV from latent
        c_kv, k_content, k_rope, v = self._compute_kv(
            x, use_cache, position_offset
        )

        # Combine content and RoPE components
        # Q: (batch, heads, seq_len, d_head + d_rope)
        q = torch.cat([q_content, q_rope], dim=-1)

        # K: (batch, heads, kv_len, d_head + d_rope)
        # K_rope is shared across heads, broadcast
        k_rope_expanded = k_rope.expand(batch, self.num_heads, -1, self.d_rope)
        k = torch.cat([k_content, k_rope_expanded], dim=-1)

        # V: (batch, heads, kv_len, d_head)
        # Note: V doesn't use RoPE component, so we pad to match Q/K
        v_padded = F.pad(v, (0, self.d_rope))

        # Compute attention
        attn_out = self.attention(q, k, v_padded, mask)

        # Remove RoPE padding from output
        attn_out = attn_out[..., : self.d_head]

        # Reshape and project output
        attn_out = attn_out.transpose(1, 2).reshape(batch, seq_len, -1)
        output = self.out_proj(attn_out)

        # Convert back to original dtype
        if output.dtype != original_dtype:
            output = output.to(original_dtype)

        return output, c_kv if use_cache else None

    def _compute_queries(
        self,
        x: torch.Tensor,
        position_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute query content and RoPE components."""
        batch, seq_len, _ = x.shape

        # Content queries: (batch, seq_len, num_heads * d_head)
        q_content = self.q_content_proj(x)
        q_content = q_content.view(batch, seq_len, self.num_heads, self.d_head)
        q_content = q_content.transpose(1, 2)  # (batch, heads, seq, d_head)

        # RoPE queries: (batch, seq_len, num_heads * d_rope)
        q_rope = self.q_rope_proj(x)
        q_rope = q_rope.view(batch, seq_len, self.num_heads, self.d_rope)
        q_rope = q_rope.transpose(1, 2)  # (batch, heads, seq, d_rope)

        # Apply RoPE to positional component
        q_rope = self.rope(q_rope, position_offset)

        return q_content, q_rope

    def _compute_kv(
        self,
        x: torch.Tensor,
        use_cache: bool,
        position_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute KV from input, using cache if available."""
        batch, seq_len, _ = x.shape

        # Compress to latent: (batch, seq_len, d_latent)
        c_kv = self.kv_down_proj(x)

        # Update cache if using
        if use_cache and self.kv_cache is not None:
            c_kv_full = self.kv_cache.update(c_kv)
        else:
            c_kv_full = c_kv

        kv_len = c_kv_full.shape[1]

        # Up-project to K content: (batch, kv_len, num_heads * d_head)
        k_content = self.k_content_up(c_kv_full)
        k_content = k_content.view(batch, kv_len, self.num_heads, self.d_head)
        k_content = k_content.transpose(1, 2)  # (batch, heads, kv_len, d_head)

        # Up-project to K RoPE: (batch, kv_len, d_rope) - shared
        k_rope = self.k_rope_up(c_kv_full)
        k_rope = k_rope.view(batch, kv_len, 1, self.d_rope)
        k_rope = k_rope.transpose(1, 2)  # (batch, 1, kv_len, d_rope)

        # Apply RoPE to K positional component
        # For cached inference, we need full RoPE positions
        if use_cache:
            k_rope = self.rope(k_rope, position_offset=0)
        else:
            k_rope = self.rope(k_rope, position_offset)

        # Up-project to V: (batch, kv_len, num_heads * d_head)
        v = self.v_up(c_kv_full)
        v = v.view(batch, kv_len, self.num_heads, self.d_head)
        v = v.transpose(1, 2)  # (batch, heads, kv_len, d_head)

        return c_kv, k_content, k_rope, v

    def memory_reduction_ratio(self) -> float:
        """Compute memory reduction ratio vs standard attention."""
        standard_kv = 2 * self.num_heads * self.d_head  # K and V
        latent = self.d_latent
        return standard_kv / latent

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, num_heads={self.num_heads}, "
            f"d_latent={self.d_latent}, d_rope={self.d_rope}, "
            f"memory_reduction={self.memory_reduction_ratio():.1f}x"
        )
