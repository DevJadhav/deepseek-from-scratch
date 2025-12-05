"""
ANE Cache Module

This module provides Apple Neural Engine optimized KV cache implementations
for efficient autoregressive generation on Apple Silicon.

Cache Components:
- UnifiedMemoryKVCache: Zero-copy KV cache exploiting unified memory
- LatentKVCache: Compressed latent cache for MLA (14x memory reduction)
- KVSplitCache: Differentiated K8V4 quantization cache
- SlidingWindowKVCache: Rolling cache for extreme sequence lengths

Usage:
    from ane_impl.cache import (
        UnifiedMemoryKVCache,
        UnifiedMemoryConfig,
        KVSplitCache,
        KVSplitConfig,
        KVQuantType,
        LatentKVCacheUnified,
        LatentCacheConfig,
        LatentQuantType,
        SlidingWindowKVCache,
        SlidingWindowConfig,
    )
"""

from .kv_split import (
    KVQuantType,
    KVSplitCache,
    KVSplitConfig,
    KVSplitQuantizer,
    QuantizedKV,
    dequantize_keys_int8,
    dequantize_values_int4,
    quantize_keys_int8,
    quantize_values_int4,
)
from .latent_cache import (
    LatentCacheConfig,
    LatentCacheStats,
    LatentKVCacheUnified,
    LatentQuantType,
)
from .sliding_window import (
    SlidingWindowConfig,
    SlidingWindowKVCache,
    SlidingWindowStats,
    StreamingSlidingWindowCache,
    WindowEvictionPolicy,
    create_sliding_window_cache,
    estimate_memory_savings,
)
from .unified_kv import (
    ComputeUnit,
    UnifiedMemoryConfig,
    UnifiedMemoryKVCache,
    UnifiedMemoryStats,
    check_unified_memory_available,
    zero_copy_transfer,
)

__all__ = [
    # Unified Memory Cache
    "UnifiedMemoryKVCache",
    "UnifiedMemoryConfig",
    "UnifiedMemoryStats",
    "ComputeUnit",
    "zero_copy_transfer",
    "check_unified_memory_available",
    # KV Split Cache
    "KVSplitCache",
    "KVSplitConfig",
    "KVQuantType",
    "KVSplitQuantizer",
    "QuantizedKV",
    "quantize_keys_int8",
    "quantize_values_int4",
    "dequantize_keys_int8",
    "dequantize_values_int4",
    # Latent Cache
    "LatentKVCacheUnified",
    "LatentCacheConfig",
    "LatentCacheStats",
    "LatentQuantType",
    # Sliding Window Cache
    "SlidingWindowKVCache",
    "SlidingWindowConfig",
    "SlidingWindowStats",
    "StreamingSlidingWindowCache",
    "WindowEvictionPolicy",
    "create_sliding_window_cache",
    "estimate_memory_savings",
]

