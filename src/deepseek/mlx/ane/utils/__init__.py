"""
ANE Utilities Module

This module provides utilities for ANE-optimized operations:
- tensor_ops: Shape normalization, layout conversion, memory alignment
- quantization: INT8/INT4 quantization and dequantization
"""

from .tensor_ops import (
    ANE_ALIGNMENT,
    ANE_MAX_TENSOR_SIZE,
    align_tensor,
    check_ane_compatible,
    convert_from_channel_last,
    convert_to_channel_last,
    get_aligned_size,
    get_ane_friendly_dim,
    is_aligned,
    nchw_to_nhwc,
    nhwc_to_nchw,
    normalize_shape_for_ane,
    pad_to_multiple,
    pad_to_power_of_2,
    split_for_ane_tiles,
    unpad_tensor,
)
from .quantization import (
    QuantizationConfig,
    QuantizationType,
    QuantizedTensor,
    compute_compression_ratio,
    compute_scale_per_block,
    compute_scale_per_channel,
    dequantize_int4_block,
    dequantize_int8_per_channel,
    get_quantized_size_bytes,
    quantize_int4_block,
    quantize_int8_per_channel,
)

__all__ = [
    # Tensor operations
    "ANE_ALIGNMENT",
    "ANE_MAX_TENSOR_SIZE",
    "align_tensor",
    "check_ane_compatible",
    "convert_from_channel_last",
    "convert_to_channel_last",
    "get_aligned_size",
    "get_ane_friendly_dim",
    "is_aligned",
    "nchw_to_nhwc",
    "nhwc_to_nchw",
    "normalize_shape_for_ane",
    "pad_to_multiple",
    "pad_to_power_of_2",
    "split_for_ane_tiles",
    "unpad_tensor",
    # Quantization
    "QuantizationConfig",
    "QuantizationType",
    "QuantizedTensor",
    "compute_compression_ratio",
    "compute_scale_per_block",
    "compute_scale_per_channel",
    "dequantize_int4_block",
    "dequantize_int8_per_channel",
    "get_quantized_size_bytes",
    "quantize_int4_block",
    "quantize_int8_per_channel",
]
