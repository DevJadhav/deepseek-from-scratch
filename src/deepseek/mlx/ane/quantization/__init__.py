"""
ANE Quantization Module

This module provides comprehensive quantization support for ANE-optimized models:

Weight Quantization:
- INT8 per-channel symmetric quantization for attention projections
- INT4 per-block quantization for FFN/expert weights
- Mixed precision strategies for critical vs non-critical layers

Activation Quantization:
- FP16 activations (default for ANE precision)
- INT8 activations for non-critical paths
- Dynamic quantization with runtime calibration

Usage:
    from ane_impl.quantization import (
        # Weight quantization
        ANEWeightQuantizer,
        WeightQuantConfig,
        QuantizedWeight,
        # Activation quantization
        ANEActivationQuantizer,
        ActivationQuantConfig,
        # Mixed precision
        MixedPrecisionConfig,
        MixedPrecisionManager,
        # Dynamic quantization
        DynamicQuantizer,
        CalibrationConfig,
    )
"""

from .weight_quant import (
    ANEWeightQuantizer,
    QuantizedWeight,
    WeightQuantConfig,
    WeightQuantType,
    quantize_weight_int8,
    quantize_weight_int4,
    dequantize_weight_int8,
    dequantize_weight_int4,
)
from .activation_quant import (
    ANEActivationQuantizer,
    ActivationQuantConfig,
    ActivationQuantType,
    QuantizedActivation,
    quantize_activation_fp16,
    quantize_activation_int8,
)
from .mixed_precision import (
    LayerPrecision,
    MixedPrecisionConfig,
    MixedPrecisionManager,
)
from .dynamic_quant import (
    CalibrationConfig,
    CalibrationStats,
    DynamicQuantizer,
)

__all__ = [
    # Weight quantization
    "ANEWeightQuantizer",
    "QuantizedWeight",
    "WeightQuantConfig",
    "WeightQuantType",
    "quantize_weight_int8",
    "quantize_weight_int4",
    "dequantize_weight_int8",
    "dequantize_weight_int4",
    # Activation quantization
    "ANEActivationQuantizer",
    "ActivationQuantConfig",
    "ActivationQuantType",
    "QuantizedActivation",
    "quantize_activation_fp16",
    "quantize_activation_int8",
    # Mixed precision
    "LayerPrecision",
    "MixedPrecisionConfig",
    "MixedPrecisionManager",
    # Dynamic quantization
    "CalibrationConfig",
    "CalibrationStats",
    "DynamicQuantizer",
]
