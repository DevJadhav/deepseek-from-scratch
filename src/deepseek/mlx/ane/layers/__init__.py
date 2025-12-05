"""
ANE-Optimized Layers Module

This module provides ANE-optimized neural network layers:
- Base layers: ANERMSNorm, ANELinear, ANEEmbedding
- Activations: ANESiLU, ANEGELU, ANESwiGLU, ANEFusedSwiGLU

All layers are optimized for Apple Neural Engine with:
- FP16 computation for activations
- INT8/INT4 weight quantization support
- Channel-last (NHWC) layout preference
- ANE-friendly dimension padding
"""

from .activations import (
    ANEFusedSwiGLU,
    ANEGELU,
    ANESiLU,
    ANESwiGLU,
)
from .base import (
    ANEEmbedding,
    ANELinear,
    ANERMSNorm,
)

__all__ = [
    # Base layers
    "ANERMSNorm",
    "ANELinear",
    "ANEEmbedding",
    # Activations
    "ANESiLU",
    "ANEGELU",
    "ANESwiGLU",
    "ANEFusedSwiGLU",
]
