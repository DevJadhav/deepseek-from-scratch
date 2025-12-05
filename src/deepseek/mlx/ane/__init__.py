"""
ANE Implementation Module

This module provides Apple Neural Engine (ANE) optimized implementations
of DeepSeek model components for efficient on-device inference on Apple Silicon.

Architecture Overview:
- config/: Model configuration classes (ANEModelConfig, TinyANEConfig, etc.)
- layers/: ANE-optimized layers (ANERMSNorm, ANELinear, ANEEmbedding, activations)
- attention/: ANE-optimized attention (MLA, KVCache, RoPE, Chunked Attention)
- moe/: ANE-optimized Mixture of Experts (Fused, Batched, Hierarchical)
- model/: Complete DeepSeek transformer model (Layer, Full Model)
- export/: CoreML export utilities for ANE deployment
- utils/: Utility functions (tensor ops, quantization)
- tests/: Unit tests for all components

Usage:
    from ane_impl.config import TinyANEConfig
    from ane_impl.layers import ANERMSNorm, ANELinear, ANEEmbedding, ANESiLU
    from ane_impl.attention import ANEMultiLatentAttention, ANEKVCache, ANERoPE
    from ane_impl.moe import ANEMoE, ANEMoEConfig, ANEExpert
    from ane_impl.model import ANEDeepSeekModel, ANEDeepSeekConfig
    from ane_impl.export import CoreMLExporter, ANEExportConfig
    from ane_impl.utils import quantize_int8_per_channel, pad_to_multiple
"""

from .config.model_config import (
    ANELayerConfig,
    ANEModelConfig,
    ANEPrecision,
    ANEQuantConfig,
    DeepSeekV3ANEConfig,
    SmallANEConfig,
    TinyANEConfig,
)
from .layers.activations import (
    ANEFusedSwiGLU,
    ANEGELU,
    ANESiLU,
    ANESwiGLU,
)
from .layers.base import (
    ANEEmbedding,
    ANELinear,
    ANERMSNorm,
)
from .attention import (
    ANEChunkedAttention,
    ANEKVCache,
    ANELatentKVCache,
    ANEMLAConfig,
    ANEMultiLatentAttention,
    ANERoPE,
    ANERoPEConfig,
    ANESlidingWindowCache,
    RoPEScalingType,
    chunked_attention_forward,
)
from .moe import (
    ANEExpert,
    ANEExpertConfig,
    ANESharedExpert,
    ANEExpertGroup,
    ANEFusedExpert,
    ANERouter,
    ANERouterConfig,
    ANEHierarchicalRouter,
    ANEMoE,
    ANEMoEConfig,
    ANEMoEFused,
    ANEMoEBatched,
    ANEMoEHierarchical,
    MoEStrategy,
    RoutingStrategy,
)
from .model import (
    ANEDeepSeekLayer,
    ANEDeepSeekLayerConfig,
    ANEDeepSeekLayerWithCache,
    ANEDeepSeekConfig,
    ANEDeepSeekModel,
    ANEGenerationConfig,
    ANEDeepSeekForCausalLM,
    create_model_from_config,
)
from .export import (
    ANEExportConfig,
    ComputeUnit,
    CoreMLOptimizationConfig,
    CoreMLExporter,
    CoreMLInference,
    validate_coreml_export,
)
from .KVcache import (
    UnifiedMemoryKVCache,
    UnifiedMemoryConfig,
    UnifiedMemoryStats,
    KVSplitCache,
    KVSplitConfig,
    KVQuantType,
    KVSplitQuantizer,
    LatentKVCacheUnified,
    LatentCacheConfig,
    LatentCacheStats,
    LatentQuantType,
    check_unified_memory_available,
    zero_copy_transfer,
)

__all__ = [
    # Config
    "ANELayerConfig",
    "ANEModelConfig",
    "ANEPrecision",
    "ANEQuantConfig",
    "TinyANEConfig",
    "SmallANEConfig",
    "DeepSeekV3ANEConfig",
    # Layers
    "ANERMSNorm",
    "ANELinear",
    "ANEEmbedding",
    "ANESiLU",
    "ANEGELU",
    "ANESwiGLU",
    "ANEFusedSwiGLU",
    # Attention
    "ANEMultiLatentAttention",
    "ANEMLAConfig",
    "ANEKVCache",
    "ANELatentKVCache",
    "ANESlidingWindowCache",
    "ANERoPE",
    "ANERoPEConfig",
    "RoPEScalingType",
    "ANEChunkedAttention",
    "chunked_attention_forward",
    # MoE
    "ANEExpert",
    "ANEExpertConfig",
    "ANESharedExpert",
    "ANEExpertGroup",
    "ANEFusedExpert",
    "ANERouter",
    "ANERouterConfig",
    "ANEHierarchicalRouter",
    "ANEMoE",
    "ANEMoEConfig",
    "ANEMoEFused",
    "ANEMoEBatched",
    "ANEMoEHierarchical",
    "MoEStrategy",
    "RoutingStrategy",
    # Model
    "ANEDeepSeekLayer",
    "ANEDeepSeekLayerConfig",
    "ANEDeepSeekLayerWithCache",
    "ANEDeepSeekConfig",
    "ANEDeepSeekModel",
    "ANEGenerationConfig",
    "ANEDeepSeekForCausalLM",
    "create_model_from_config",
    # Export
    "ANEExportConfig",
    "ComputeUnit",
    "CoreMLOptimizationConfig",
    "CoreMLExporter",
    "CoreMLInference",
    "validate_coreml_export",
    # Cache (Phase 5)
    "UnifiedMemoryKVCache",
    "UnifiedMemoryConfig",
    "UnifiedMemoryStats",
    "KVSplitCache",
    "KVSplitConfig",
    "KVQuantType",
    "KVSplitQuantizer",
    "LatentKVCacheUnified",
    "LatentCacheConfig",
    "LatentCacheStats",
    "LatentQuantType",
    "check_unified_memory_available",
    "zero_copy_transfer",
]
