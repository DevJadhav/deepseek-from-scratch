"""
ANE Model Module

This module provides the complete DeepSeek transformer model optimized for
Apple Neural Engine (ANE) inference on Apple Silicon.

Components:
- ANEDeepSeekConfig: Configuration dataclass for the full model
- ANEDeepSeekLayer: Single transformer layer with ANE-optimized attention + MoE
- ANEDeepSeekModel: Complete transformer model with embeddings, layers, and output head
- ANEGenerationConfig: Configuration for text generation

Usage:
    from ane_impl.model import (
        ANEDeepSeekConfig,
        ANEDeepSeekLayer,
        ANEDeepSeekModel,
        ANEGenerationConfig,
    )
    
    # Create a tiny config for testing
    config = ANEDeepSeekConfig.tiny()
    model = ANEDeepSeekModel(config)
    
    # Forward pass
    input_ids = torch.randint(0, config.vocab_size, (1, 128))
    logits = model(input_ids)
"""

from .layer import (
    ANEDeepSeekLayer,
    ANEDeepSeekLayerConfig,
    ANEDeepSeekLayerWithCache,
)
from .transformer import (
    ANEDeepSeekConfig,
    ANEDeepSeekModel,
    ANEDeepSeekForCausalLM,
    ANEGenerationConfig,
    create_model_from_config,
)

__all__ = [
    # Layer
    "ANEDeepSeekLayer",
    "ANEDeepSeekLayerConfig",
    "ANEDeepSeekLayerWithCache",
    # Transformer Model
    "ANEDeepSeekConfig",
    "ANEDeepSeekModel",
    "ANEDeepSeekForCausalLM",
    "ANEGenerationConfig",
    "create_model_from_config",
]
