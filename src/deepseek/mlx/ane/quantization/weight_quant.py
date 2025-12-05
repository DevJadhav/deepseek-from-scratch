"""
ANE Weight Quantization Module

This module provides weight quantization optimized for Apple Neural Engine:
- INT8 per-channel symmetric quantization for attention projections
- INT4 per-block (group_size=128) quantization for FFN/expert weights
- Automatic scale computation and management
- Efficient packing for INT4 weights

Strategy (as per production_hardening.md):
- Attention projections: INT8 per-channel
- FFN/Expert weights: INT4 block-wise (group_size=128)
- Embedding: FP16 (CPU-side)
- LayerNorm weights: FP16
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn


class WeightQuantType(Enum):
    """Weight quantization types for ANE."""
    
    NONE = "none"  # FP16 (no quantization)
    INT8_PER_CHANNEL = "int8_per_channel"  # Per-channel symmetric INT8
    INT8_PER_TENSOR = "int8_per_tensor"  # Per-tensor symmetric INT8
    INT4_PER_BLOCK = "int4_per_block"  # Per-block INT4 (group_size=128)
    INT4_PER_CHANNEL = "int4_per_channel"  # Per-channel INT4


@dataclass
class WeightQuantConfig:
    """Configuration for weight quantization."""
    
    # Quantization type
    quant_type: WeightQuantType = WeightQuantType.INT8_PER_CHANNEL
    
    # INT4 block/group size
    block_size: int = 128
    
    # Symmetric vs asymmetric
    symmetric: bool = True
    
    # Use absmax for scale computation
    use_absmax: bool = True
    
    # Clamp outliers before quantization (percentile)
    clamp_outliers: bool = False
    outlier_percentile: float = 99.9
    
    # Store original weights for fine-tuning
    store_original: bool = False


@dataclass
class QuantizedWeight:
    """Container for quantized weight tensor."""
    
    # Quantized data (INT8 or packed INT4)
    data: torch.Tensor
    
    # Scale factors
    scale: torch.Tensor
    
    # Zero point (None for symmetric quantization)
    zero_point: torch.Tensor | None
    
    # Original tensor shape
    original_shape: tuple[int, ...]
    
    # Quantization configuration
    config: WeightQuantConfig
    
    # Original weights (optional, for fine-tuning)
    original_weights: torch.Tensor | None = None
    
    @property
    def dtype(self) -> torch.dtype:
        """Get the quantized data type."""
        return self.data.dtype
    
    @property
    def is_symmetric(self) -> bool:
        """Check if quantization is symmetric."""
        return self.zero_point is None
    
    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio vs FP16."""
        original_bytes = math.prod(self.original_shape) * 2  # FP16
        
        quant_bytes = self.data.numel() * self.data.element_size()
        scale_bytes = self.scale.numel() * self.scale.element_size()
        zp_bytes = self.zero_point.numel() * self.zero_point.element_size() if self.zero_point is not None else 0
        
        total_quant_bytes = quant_bytes + scale_bytes + zp_bytes
        return original_bytes / total_quant_bytes if total_quant_bytes > 0 else 1.0
    
    def memory_bytes(self) -> int:
        """Get total memory usage in bytes."""
        total = self.data.numel() * self.data.element_size()
        total += self.scale.numel() * self.scale.element_size()
        if self.zero_point is not None:
            total += self.zero_point.numel() * self.zero_point.element_size()
        return total


def quantize_weight_int8(
    weight: torch.Tensor,
    axis: int = 0,
    symmetric: bool = True,
    config: WeightQuantConfig | None = None,
) -> QuantizedWeight:
    """
    Quantize weights to INT8 with per-channel scaling.
    
    Per-channel symmetric quantization:
        scale[c] = max(|W[c]|) / 127
        W_quant[c] = round(W[c] / scale[c])
    
    Args:
        weight: Weight tensor (typically [out_features, in_features])
        axis: Channel axis for per-channel quantization (default 0)
        symmetric: Use symmetric quantization (default True)
        config: Optional quantization configuration
        
    Returns:
        QuantizedWeight containing INT8 data and scale
    """
    if config is None:
        config = WeightQuantConfig(
            quant_type=WeightQuantType.INT8_PER_CHANNEL,
            symmetric=symmetric,
        )
    
    original_shape = weight.shape
    device = weight.device
    
    # Clamp outliers if requested
    if config.clamp_outliers:
        percentile = config.outlier_percentile
        lower = torch.quantile(weight.float().flatten(), (100 - percentile) / 100)
        upper = torch.quantile(weight.float().flatten(), percentile / 100)
        weight = weight.clamp(lower, upper)
    
    # Compute per-channel scale
    if axis != 0:
        weight = weight.transpose(0, axis)
    
    weight_2d = weight.reshape(weight.shape[0], -1)
    
    if symmetric:
        # Symmetric: scale based on max absolute value
        qmax = 127  # INT8 max for symmetric
        abs_max = weight_2d.abs().max(dim=1).values
        scale = abs_max / qmax
        scale = torch.clamp(scale, min=1e-10)
        zero_point = None
        
        # Quantize
        scale_expanded = scale.unsqueeze(1)
        weight_quant = torch.round(weight_2d / scale_expanded).clamp(-128, 127).to(torch.int8)
    else:
        # Asymmetric: scale based on min/max range
        qmax = 255  # INT8 max for asymmetric
        w_min = weight_2d.min(dim=1).values
        w_max = weight_2d.max(dim=1).values
        scale = (w_max - w_min) / qmax
        scale = torch.clamp(scale, min=1e-10)
        zero_point = torch.round(-w_min / scale).to(torch.int8)
        
        # Quantize
        scale_expanded = scale.unsqueeze(1)
        zp_expanded = zero_point.float().unsqueeze(1)
        weight_quant = torch.round(weight_2d / scale_expanded + zp_expanded).clamp(0, 255).to(torch.uint8)
    
    # Reshape back
    weight_quant = weight_quant.reshape(weight.shape)
    if axis != 0:
        weight_quant = weight_quant.transpose(0, axis)
    
    return QuantizedWeight(
        data=weight_quant.to(device),
        scale=scale.to(device),
        zero_point=zero_point.to(device) if zero_point is not None else None,
        original_shape=original_shape,
        config=config,
        original_weights=weight.clone() if config.store_original else None,
    )


def dequantize_weight_int8(
    quantized: QuantizedWeight,
    axis: int = 0,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Dequantize INT8 weights to floating point.
    
    For symmetric:
        W_dequant = W_quant * scale
    
    For asymmetric:
        W_dequant = (W_quant - zero_point) * scale
    
    Args:
        quantized: QuantizedWeight from quantize_weight_int8
        axis: Channel axis (default 0)
        output_dtype: Output data type (default FP16)
        
    Returns:
        Dequantized weight tensor
    """
    weight_quant = quantized.data.float()
    scale = quantized.scale
    
    # Handle axis
    if axis != 0:
        weight_quant = weight_quant.transpose(0, axis)
    
    weight_2d = weight_quant.reshape(weight_quant.shape[0], -1)
    
    if quantized.is_symmetric:
        scale_expanded = scale.unsqueeze(1)
        weight_dequant = weight_2d * scale_expanded
    else:
        scale_expanded = scale.unsqueeze(1)
        zp_expanded = quantized.zero_point.float().unsqueeze(1)
        weight_dequant = (weight_2d - zp_expanded) * scale_expanded
    
    # Reshape back
    weight_dequant = weight_dequant.reshape(weight_quant.shape)
    if axis != 0:
        weight_dequant = weight_dequant.transpose(0, axis)
    
    return weight_dequant.to(output_dtype)


def quantize_weight_int4(
    weight: torch.Tensor,
    block_size: int = 128,
    symmetric: bool = True,
    config: WeightQuantConfig | None = None,
) -> QuantizedWeight:
    """
    Quantize weights to INT4 with per-block scaling.
    
    Per-block (group) quantization for FFN/expert weights:
        - Divide tensor into blocks of size block_size
        - Compute scale per block
        - Quantize to 4-bit values
        - Pack 2 values per byte
    
    Args:
        weight: Weight tensor (2D: [out_features, in_features])
        block_size: Block/group size for scaling (default 128)
        symmetric: Use symmetric quantization (default True)
        config: Optional quantization configuration
        
    Returns:
        QuantizedWeight containing packed INT4 data and scale
    """
    if config is None:
        config = WeightQuantConfig(
            quant_type=WeightQuantType.INT4_PER_BLOCK,
            block_size=block_size,
            symmetric=symmetric,
        )
    
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D tensor for INT4 block quantization, got {weight.ndim}D")
    
    original_shape = weight.shape
    device = weight.device
    rows, cols = weight.shape
    
    # Calculate number of blocks
    num_row_blocks = math.ceil(rows / block_size)
    num_col_blocks = math.ceil(cols / block_size)
    
    # Pad tensor if needed
    padded_rows = num_row_blocks * block_size
    padded_cols = num_col_blocks * block_size
    
    if rows != padded_rows or cols != padded_cols:
        weight_padded = torch.nn.functional.pad(
            weight, (0, padded_cols - cols, 0, padded_rows - rows)
        )
    else:
        weight_padded = weight
    
    # Reshape into blocks: (num_row_blocks, block_size, num_col_blocks, block_size)
    weight_blocks = weight_padded.reshape(
        num_row_blocks, block_size, num_col_blocks, block_size
    )
    # Transpose to (num_row_blocks, num_col_blocks, block_size, block_size)
    weight_blocks = weight_blocks.permute(0, 2, 1, 3)
    
    # Compute per-block scale
    weight_blocks_flat = weight_blocks.reshape(num_row_blocks, num_col_blocks, -1)
    
    if symmetric:
        qmax = 7  # INT4 symmetric max
        abs_max = weight_blocks_flat.abs().max(dim=2).values
        scale = abs_max / qmax
        scale = torch.clamp(scale, min=1e-10)
        zero_point = None
        
        # Quantize to INT4 range [-8, 7]
        scale_expanded = scale.unsqueeze(-1).unsqueeze(-1)
        weight_quant = torch.round(weight_blocks / scale_expanded).clamp(-8, 7).to(torch.int8)
    else:
        qmax = 15  # INT4 asymmetric max
        w_min = weight_blocks_flat.min(dim=2).values
        w_max = weight_blocks_flat.max(dim=2).values
        scale = (w_max - w_min) / qmax
        scale = torch.clamp(scale, min=1e-10)
        zero_point = torch.round(-w_min / scale).to(torch.int8)
        
        # Quantize to INT4 range [0, 15]
        scale_expanded = scale.unsqueeze(-1).unsqueeze(-1)
        zp_expanded = zero_point.float().unsqueeze(-1).unsqueeze(-1)
        weight_quant = torch.round(weight_blocks / scale_expanded + zp_expanded).clamp(0, 15).to(torch.int8)
    
    # Pack INT4 into INT8 (2 values per byte)
    weight_quant_flat = weight_quant.reshape(-1)
    
    # Ensure even number of elements
    if weight_quant_flat.numel() % 2 != 0:
        weight_quant_flat = torch.nn.functional.pad(weight_quant_flat, (0, 1))
    
    # Pack: low nibble = even indices, high nibble = odd indices
    packed = (weight_quant_flat[0::2] & 0x0F) | ((weight_quant_flat[1::2] & 0x0F) << 4)
    packed = packed.to(torch.int8)
    
    return QuantizedWeight(
        data=packed.to(device),
        scale=scale.to(device),
        zero_point=zero_point.to(device) if zero_point is not None else None,
        original_shape=original_shape,
        config=config,
        original_weights=weight.clone() if config.store_original else None,
    )


def dequantize_weight_int4(
    quantized: QuantizedWeight,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Dequantize INT4 weights to floating point.
    
    Args:
        quantized: QuantizedWeight from quantize_weight_int4
        output_dtype: Output data type (default FP16)
        
    Returns:
        Dequantized weight tensor
    """
    packed = quantized.data
    scale = quantized.scale
    config = quantized.config
    original_shape = quantized.original_shape
    block_size = config.block_size
    
    rows, cols = original_shape
    num_row_blocks = math.ceil(rows / block_size)
    num_col_blocks = math.ceil(cols / block_size)
    
    # Unpack INT4 from INT8
    low_nibble = (packed & 0x0F).to(torch.int8)
    high_nibble = ((packed >> 4) & 0x0F).to(torch.int8)
    
    # Handle signed INT4 (sign extend)
    if quantized.is_symmetric:
        low_nibble = torch.where(low_nibble > 7, low_nibble - 16, low_nibble)
        high_nibble = torch.where(high_nibble > 7, high_nibble - 16, high_nibble)
    
    # Interleave to restore original order
    unpacked = torch.zeros(packed.numel() * 2, dtype=torch.float32, device=packed.device)
    unpacked[0::2] = low_nibble.float()
    unpacked[1::2] = high_nibble.float()
    
    # Reshape to blocks
    padded_rows = num_row_blocks * block_size
    padded_cols = num_col_blocks * block_size
    total_elements = padded_rows * padded_cols
    
    weight_blocks = unpacked[:total_elements].reshape(
        num_row_blocks, num_col_blocks, block_size, block_size
    )
    
    # Apply scale
    scale_expanded = scale.unsqueeze(-1).unsqueeze(-1)
    
    if quantized.is_symmetric:
        weight_dequant = weight_blocks * scale_expanded
    else:
        zp_expanded = quantized.zero_point.float().unsqueeze(-1).unsqueeze(-1)
        weight_dequant = (weight_blocks - zp_expanded) * scale_expanded
    
    # Reshape back: (num_row_blocks, num_col_blocks, block_size, block_size)
    # -> (num_row_blocks, block_size, num_col_blocks, block_size)
    # -> (padded_rows, padded_cols)
    weight_dequant = weight_dequant.permute(0, 2, 1, 3).reshape(padded_rows, padded_cols)
    
    # Remove padding
    weight_dequant = weight_dequant[:rows, :cols]
    
    return weight_dequant.to(output_dtype)


class ANEWeightQuantizer(nn.Module):
    """
    ANE-optimized weight quantizer for model weights.
    
    This quantizer applies appropriate quantization strategies:
    - Attention projections: INT8 per-channel
    - FFN/Expert weights: INT4 block-wise (group_size=128)
    - Embedding: FP16 (CPU-side, no quantization)
    - LayerNorm weights: FP16 (no quantization)
    
    Example:
        quantizer = ANEWeightQuantizer(
            default_config=WeightQuantConfig(quant_type=WeightQuantType.INT8_PER_CHANNEL),
        )
        
        # Quantize a weight tensor
        quant_weight = quantizer.quantize(model.attention.q_proj.weight)
        
        # Dequantize for inference
        weight = quantizer.dequantize(quant_weight)
    """
    
    def __init__(
        self,
        default_config: WeightQuantConfig | None = None,
        attention_config: WeightQuantConfig | None = None,
        ffn_config: WeightQuantConfig | None = None,
    ):
        """
        Initialize weight quantizer.
        
        Args:
            default_config: Default quantization config
            attention_config: Config for attention layers (default: INT8 per-channel)
            ffn_config: Config for FFN layers (default: INT4 per-block)
        """
        super().__init__()
        
        self.default_config = default_config or WeightQuantConfig(
            quant_type=WeightQuantType.INT8_PER_CHANNEL,
        )
        
        self.attention_config = attention_config or WeightQuantConfig(
            quant_type=WeightQuantType.INT8_PER_CHANNEL,
            symmetric=True,
        )
        
        self.ffn_config = ffn_config or WeightQuantConfig(
            quant_type=WeightQuantType.INT4_PER_BLOCK,
            block_size=128,
            symmetric=True,
        )
        
        self._quantized_weights: dict[str, QuantizedWeight] = {}
    
    def quantize(
        self,
        weight: torch.Tensor,
        name: str | None = None,
        config: WeightQuantConfig | None = None,
    ) -> QuantizedWeight:
        """
        Quantize a weight tensor.
        
        Args:
            weight: Weight tensor to quantize
            name: Optional name for caching
            config: Optional config (uses default if None)
            
        Returns:
            QuantizedWeight
        """
        if config is None:
            config = self.default_config
        
        if config.quant_type == WeightQuantType.NONE:
            return QuantizedWeight(
                data=weight.to(torch.float16),
                scale=torch.ones(1, device=weight.device),
                zero_point=None,
                original_shape=weight.shape,
                config=config,
            )
        elif config.quant_type in (WeightQuantType.INT8_PER_CHANNEL, WeightQuantType.INT8_PER_TENSOR):
            result = quantize_weight_int8(weight, config=config)
        elif config.quant_type in (WeightQuantType.INT4_PER_BLOCK, WeightQuantType.INT4_PER_CHANNEL):
            result = quantize_weight_int4(weight, config=config)
        else:
            raise ValueError(f"Unsupported quantization type: {config.quant_type}")
        
        if name is not None:
            self._quantized_weights[name] = result
        
        return result
    
    def dequantize(
        self,
        quantized: QuantizedWeight,
        output_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """
        Dequantize a weight tensor.
        
        Args:
            quantized: QuantizedWeight to dequantize
            output_dtype: Output data type (default FP16)
            
        Returns:
            Dequantized weight tensor
        """
        config = quantized.config
        
        if config.quant_type == WeightQuantType.NONE:
            return quantized.data.to(output_dtype)
        elif config.quant_type in (WeightQuantType.INT8_PER_CHANNEL, WeightQuantType.INT8_PER_TENSOR):
            return dequantize_weight_int8(quantized, output_dtype=output_dtype)
        elif config.quant_type in (WeightQuantType.INT4_PER_BLOCK, WeightQuantType.INT4_PER_CHANNEL):
            return dequantize_weight_int4(quantized, output_dtype=output_dtype)
        else:
            raise ValueError(f"Unsupported quantization type: {config.quant_type}")
    
    def quantize_attention(self, weight: torch.Tensor, name: str | None = None) -> QuantizedWeight:
        """Quantize attention weight (INT8 per-channel)."""
        return self.quantize(weight, name=name, config=self.attention_config)
    
    def quantize_ffn(self, weight: torch.Tensor, name: str | None = None) -> QuantizedWeight:
        """Quantize FFN weight (INT4 per-block)."""
        return self.quantize(weight, name=name, config=self.ffn_config)
    
    def get_total_memory_bytes(self) -> int:
        """Get total memory usage of all quantized weights."""
        return sum(qw.memory_bytes() for qw in self._quantized_weights.values())
    
    def get_compression_stats(self) -> dict:
        """Get compression statistics for all quantized weights."""
        stats = {
            "num_weights": len(self._quantized_weights),
            "total_original_bytes": 0,
            "total_quantized_bytes": 0,
            "by_type": {},
        }
        
        for _name, qw in self._quantized_weights.items():
            original_bytes = math.prod(qw.original_shape) * 2  # FP16
            quant_bytes = qw.memory_bytes()
            
            stats["total_original_bytes"] += original_bytes
            stats["total_quantized_bytes"] += quant_bytes
            
            qtype = qw.config.quant_type.value
            if qtype not in stats["by_type"]:
                stats["by_type"][qtype] = {"count": 0, "original_bytes": 0, "quant_bytes": 0}
            stats["by_type"][qtype]["count"] += 1
            stats["by_type"][qtype]["original_bytes"] += original_bytes
            stats["by_type"][qtype]["quant_bytes"] += quant_bytes
        
        if stats["total_quantized_bytes"] > 0:
            stats["compression_ratio"] = stats["total_original_bytes"] / stats["total_quantized_bytes"]
        else:
            stats["compression_ratio"] = 1.0
        
        return stats
