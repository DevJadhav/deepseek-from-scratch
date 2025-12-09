"""DeepSeek Model Components

This module provides the core model components for DeepSeek:
- Attention mechanisms (MQA, GQA, MLA) with Flash Attention support
- Transformer architecture with gradient checkpointing
- Mixture of Experts (MoE) layers
"""

import torch
from typing import Optional

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


def build_model_for_training(
    hidden_size: int = 256,
    num_layers: int = 4,
    num_attention_heads: int = 4,
    vocab_size: int = 32000,
    intermediate_size: Optional[int] = None,
    num_experts: int = 8,
    top_k: int = 2,
    use_moe: bool = False,
    device: Optional[torch.device] = None,
) -> DeepSeekModel:
    """
    Build a DeepSeek model configured for distributed training.
    
    This factory function creates a model with sensible defaults for
    training on Modal A100-40GB GPUs with 5D parallelism.
    
    Args:
        hidden_size: Model hidden dimension
        num_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        vocab_size: Vocabulary size
        intermediate_size: FFN intermediate dimension (default: 4 * hidden_size)
        num_experts: Number of MoE experts
        top_k: Number of experts to route each token to
        use_moe: Whether to use MoE layers
        device: Device to place model on
        
    Returns:
        Configured DeepSeekModel ready for training
    """
    if intermediate_size is None:
        intermediate_size = 4 * hidden_size
    
    # Calculate latent and rope dimensions based on model size
    d_latent = hidden_size // 4
    d_rope = hidden_size // num_attention_heads // 2
    
    model = DeepSeekModel(
        vocab_size=vocab_size,
        num_layers=num_layers,
        d_model=hidden_size,
        num_heads=num_attention_heads,
        d_latent=d_latent,
        d_rope=d_rope,
        d_hidden=intermediate_size,
        num_experts=num_experts,
        num_shared=1,
        num_routed=num_experts,
        top_k=top_k,
        use_moe=use_moe,
        dropout=0.0,
        checkpoint_config=GradientCheckpointConfig(
            enabled=True,
            checkpoint_every_n_layers=1,
            checkpoint_moe=True,
            use_reentrant=False,
        ),
    )
    
    if device is not None:
        model = model.to(device)
    
    return model

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
    # Factory
    "build_model_for_training",
]
