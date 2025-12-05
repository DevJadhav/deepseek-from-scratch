"""DeepSeek Model Components

This module provides the core model components for DeepSeek:
- Attention mechanisms (MQA, GQA, MLA) with Flash Attention support
- Transformer architecture with gradient checkpointing
- Mixture of Experts (MoE) layers
"""

from .attention import (
    AttentionBackend,
    FlashAttentionConfig,
    MultiQueryAttention,
    GroupedQueryAttention,
    detect_flash_attention_version,
    get_optimal_attention_backend,
    scaled_dot_product_attention_with_backend,
    chunked_attention,
)
from .mla import (
    KVCache,
    RotaryPositionalEncoding,
    ExtendedRoPEConfig,
    ExtendedRotaryPositionalEncoding,
    MultiHeadLatentAttention,
    DeepSeekAttention,
)
from .transformer import (
    RMSNorm,
    GradientCheckpointConfig,
    DeepSeekLayer,
    DeepSeekModel,
)
from .moe import (
    Expert,
    DeepSeekMoE,
    StandardMoE,
)

__all__ = [
    # Attention
    "AttentionBackend",
    "FlashAttentionConfig",
    "MultiQueryAttention",
    "GroupedQueryAttention",
    "detect_flash_attention_version",
    "get_optimal_attention_backend",
    "scaled_dot_product_attention_with_backend",
    "chunked_attention",
    # MLA
    "KVCache",
    "RotaryPositionalEncoding",
    "ExtendedRoPEConfig",
    "ExtendedRotaryPositionalEncoding",
    "MultiHeadLatentAttention",
    "DeepSeekAttention",
    # Transformer
    "RMSNorm",
    "GradientCheckpointConfig",
    "DeepSeekLayer",
    "DeepSeekModel",
    # MoE
    "Expert",
    "DeepSeekMoE",
    "StandardMoE",
]
