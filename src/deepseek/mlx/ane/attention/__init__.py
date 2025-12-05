"""
ANE-Optimized Attention Module

This module provides Apple Neural Engine optimized attention implementations:
- ANEKVCache: Zero-copy KV cache for unified memory
- ANELatentKVCache: Compressed latent KV cache for MLA
- ANERoPE: Decoupled positional encoding with NTK/YaRN scaling
- ANEChunkedAttention: 128-token window attention for ANE constraints
- ANEMultiLatentAttention: Core MLA implementation for ANE
"""

from .kv_cache import ANEKVCache, ANELatentKVCache, ANESlidingWindowCache
from .rope import ANERoPE, ANERoPEConfig, RoPEScalingType
from .chunked_attention import ANEChunkedAttention, chunked_attention_forward
from .mla import ANEMultiLatentAttention, ANEMLAConfig

__all__ = [
    # KV Cache
    "ANEKVCache",
    "ANELatentKVCache",
    "ANESlidingWindowCache",
    # RoPE
    "ANERoPE",
    "ANERoPEConfig",
    "RoPEScalingType",
    # Chunked Attention
    "ANEChunkedAttention",
    "chunked_attention_forward",
    # MLA
    "ANEMultiLatentAttention",
    "ANEMLAConfig",
]
