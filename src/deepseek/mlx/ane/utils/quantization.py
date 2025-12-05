"""
ANE Quantization Utilities

This module provides INT8/INT4 quantization and dequantization functions
optimized for ANE inference:
- Per-channel symmetric INT8 quantization
- Per-block asymmetric INT4 quantization
- Scale computation utilities
- Compression ratio calculation

ANE supports INT8 and INT4 weight quantization with FP16 activations.
Per-channel quantization provides better accuracy for attention projections.
Per-block (group) quantization is more efficient for FFN/expert weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import torch


class QuantizationType(Enum):
    """Quantization types supported for ANE."""
    
    NONE = "none"
    INT8_PER_CHANNEL = "int8_per_channel"
    INT8_PER_TENSOR = "int8_per_tensor"
    INT4_PER_BLOCK = "int4_per_block"
    INT4_PER_CHANNEL = "int4_per_channel"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"


@dataclass
class QuantizationConfig:
    """Configuration for quantization."""
    
    quant_type: QuantizationType = QuantizationType.INT8_PER_CHANNEL
    block_size: int = 128  # Block size for per-block quantization
    symmetric: bool = True  # Use symmetric quantization
    use_absmax: bool = True  # Use absmax for scale computation


@dataclass
class QuantizedTensor:
    """Container for quantized tensor data."""
    
    data: torch.Tensor  # Quantized data (int8 or packed int4)
    scale: torch.Tensor  # Scale factor(s)
    zero_point: torch.Tensor | None  # Zero point (None for symmetric)
    original_shape: tuple[int, ...]  # Original tensor shape
    quant_type: QuantizationType  # Type of quantization used
    
    @property
    def dtype(self) -> torch.dtype:
        """Get the quantized data type."""
        return self.data.dtype
    
    @property
    def is_symmetric(self) -> bool:
        """Check if quantization is symmetric."""
        return self.zero_point is None


def compute_scale_per_channel(
    x: torch.Tensor,
    axis: int = 0,
    symmetric: bool = True,
    num_bits: int = 8,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Compute quantization scale per channel.
    
    For symmetric quantization:
        scale = max(|x|) / (2^(bits-1) - 1)
        
    For asymmetric quantization:
        scale = (max(x) - min(x)) / (2^bits - 1)
        zero_point = round(-min(x) / scale)
    
    Args:
        x: Input tensor
        axis: Channel axis (default 0)
        symmetric: Use symmetric quantization (default True)
        num_bits: Number of bits (default 8)
        
    Returns:
        Tuple of (scale, zero_point). zero_point is None for symmetric.
    """
    # Move axis to first dimension for easier processing
    if axis != 0:
        x = x.transpose(0, axis)
    
    # Flatten all dims except the first (channel dim)
    x_flat = x.reshape(x.shape[0], -1)
    
    if symmetric:
        # Symmetric: scale based on max absolute value
        qmax = (1 << (num_bits - 1)) - 1  # 127 for INT8
        abs_max = x_flat.abs().max(dim=1).values
        scale = abs_max / qmax
        # Avoid division by zero
        scale = torch.clamp(scale, min=1e-10)
        return scale, None
    else:
        # Asymmetric: scale based on min/max range
        qmax = (1 << num_bits) - 1  # 255 for INT8
        x_min = x_flat.min(dim=1).values
        x_max = x_flat.max(dim=1).values
        scale = (x_max - x_min) / qmax
        scale = torch.clamp(scale, min=1e-10)
        zero_point = torch.round(-x_min / scale).to(torch.int8)
        return scale, zero_point


def compute_scale_per_block(
    x: torch.Tensor,
    block_size: int = 128,
    symmetric: bool = True,
    num_bits: int = 4,
) -> torch.Tensor:
    """
    Compute quantization scale per block.
    
    Divides the tensor into blocks of size block_size and computes
    scale for each block independently.
    
    Args:
        x: Input tensor (2D: out_features x in_features)
        block_size: Size of each block (default 128)
        symmetric: Use symmetric quantization (default True)
        num_bits: Number of bits (default 4)
        
    Returns:
        Scale tensor of shape (num_row_blocks, num_col_blocks)
    """
    if x.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got {x.ndim}D")
    
    rows, cols = x.shape
    
    # Calculate number of blocks (with padding if needed)
    num_row_blocks = math.ceil(rows / block_size)
    num_col_blocks = math.ceil(cols / block_size)
    
    # Pad tensor if needed
    pad_rows = num_row_blocks * block_size - rows
    pad_cols = num_col_blocks * block_size - cols
    
    if pad_rows > 0 or pad_cols > 0:
        x = torch.nn.functional.pad(x, (0, pad_cols, 0, pad_rows))
    
    # Reshape into blocks
    x_blocks = x.reshape(num_row_blocks, block_size, num_col_blocks, block_size)
    x_blocks = x_blocks.permute(0, 2, 1, 3).reshape(
        num_row_blocks, num_col_blocks, -1
    )
    
    if symmetric:
        qmax = (1 << (num_bits - 1)) - 1  # 7 for INT4
        abs_max = x_blocks.abs().max(dim=2).values
        scale = abs_max / qmax
    else:
        qmax = (1 << num_bits) - 1  # 15 for INT4
        x_min = x_blocks.min(dim=2).values
        x_max = x_blocks.max(dim=2).values
        scale = (x_max - x_min) / qmax
    
    scale = torch.clamp(scale, min=1e-10)
    return scale


def quantize_int8_per_channel(
    x: torch.Tensor,
    axis: int = 0,
    symmetric: bool = True,
) -> QuantizedTensor:
    """
    Quantize tensor to INT8 with per-channel scaling.
    
    Args:
        x: Input tensor (FP16/FP32)
        axis: Channel axis (default 0)
        symmetric: Use symmetric quantization (default True)
        
    Returns:
        QuantizedTensor containing INT8 data and scale
    """
    original_shape = x.shape
    
    # Compute scale
    scale, zero_point = compute_scale_per_channel(x, axis, symmetric, num_bits=8)
    
    # Reshape scale for broadcasting
    scale_shape = [1] * x.ndim
    scale_shape[axis] = x.shape[axis]
    scale_broadcast = scale.reshape(scale_shape)
    
    # Quantize
    if symmetric:
        x_quant = torch.round(x / scale_broadcast).to(torch.int8)
    else:
        zp_shape = [1] * x.ndim
        zp_shape[axis] = x.shape[axis]
        zp_broadcast = zero_point.reshape(zp_shape)
        x_quant = torch.round(x / scale_broadcast + zp_broadcast.float()).to(torch.int8)
    
    return QuantizedTensor(
        data=x_quant,
        scale=scale,
        zero_point=zero_point,
        original_shape=original_shape,
        quant_type=QuantizationType.INT8_PER_CHANNEL,
    )


def dequantize_int8_per_channel(
    quantized: QuantizedTensor,
    axis: int = 0,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Dequantize INT8 tensor with per-channel scaling.
    
    Args:
        quantized: QuantizedTensor from quantize_int8_per_channel
        axis: Channel axis (default 0)
        output_dtype: Output data type (default FP16)
        
    Returns:
        Dequantized tensor
    """
    x_quant = quantized.data.float()
    scale = quantized.scale
    
    # Reshape scale for broadcasting
    scale_shape = [1] * x_quant.ndim
    scale_shape[axis] = scale.shape[0]
    scale_broadcast = scale.reshape(scale_shape)
    
    if quantized.zero_point is not None:
        zp_shape = [1] * x_quant.ndim
        zp_shape[axis] = quantized.zero_point.shape[0]
        zp_broadcast = quantized.zero_point.float().reshape(zp_shape)
        x_dequant = (x_quant - zp_broadcast) * scale_broadcast
    else:
        x_dequant = x_quant * scale_broadcast
    
    return x_dequant.to(output_dtype)


def quantize_int4_block(
    x: torch.Tensor,
    block_size: int = 128,
    symmetric: bool = True,
) -> QuantizedTensor:
    """
    Quantize tensor to INT4 with per-block scaling.
    
    INT4 values are packed into INT8 storage (2 values per byte).
    
    Args:
        x: Input tensor (2D: out_features x in_features)
        block_size: Block size for scaling (default 128)
        symmetric: Use symmetric quantization (default True)
        
    Returns:
        QuantizedTensor containing packed INT4 data and scale
    """
    if x.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got {x.ndim}D")
    
    original_shape = x.shape
    rows, cols = x.shape
    
    # Compute per-block scale
    scale = compute_scale_per_block(x, block_size, symmetric, num_bits=4)
    num_row_blocks, num_col_blocks = scale.shape
    
    # Pad tensor to block size
    padded_rows = num_row_blocks * block_size
    padded_cols = num_col_blocks * block_size
    
    if rows != padded_rows or cols != padded_cols:
        x_padded = torch.nn.functional.pad(
            x, (0, padded_cols - cols, 0, padded_rows - rows)
        )
    else:
        x_padded = x
    
    # Reshape into blocks
    x_blocks = x_padded.reshape(num_row_blocks, block_size, num_col_blocks, block_size)
    x_blocks = x_blocks.permute(0, 2, 1, 3)  # (row_blocks, col_blocks, block_h, block_w)
    
    # Expand scale for broadcasting
    scale_expanded = scale.unsqueeze(-1).unsqueeze(-1)
    
    # Quantize to INT4 range [-8, 7] for symmetric, [0, 15] for asymmetric
    if symmetric:
        x_quant = torch.round(x_blocks / scale_expanded).clamp(-8, 7).to(torch.int8)
    else:
        x_quant = torch.round(x_blocks / scale_expanded).clamp(0, 15).to(torch.int8)
    
    # Pack INT4 into INT8 (2 values per byte)
    x_quant_flat = x_quant.reshape(-1)
    if x_quant_flat.numel() % 2 != 0:
        x_quant_flat = torch.nn.functional.pad(x_quant_flat, (0, 1))
    
    # Pack: low nibble = even indices, high nibble = odd indices
    x_packed = (x_quant_flat[0::2] & 0x0F) | ((x_quant_flat[1::2] & 0x0F) << 4)
    x_packed = x_packed.to(torch.int8)
    
    return QuantizedTensor(
        data=x_packed,
        scale=scale,
        zero_point=None,  # Zero point not used for INT4 in this implementation
        original_shape=original_shape,
        quant_type=QuantizationType.INT4_PER_BLOCK,
    )


def dequantize_int4_block(
    quantized: QuantizedTensor,
    block_size: int = 128,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Dequantize INT4 tensor with per-block scaling.
    
    Args:
        quantized: QuantizedTensor from quantize_int4_block
        block_size: Block size used for quantization
        output_dtype: Output data type (default FP16)
        
    Returns:
        Dequantized tensor
    """
    x_packed = quantized.data
    scale = quantized.scale
    original_shape = quantized.original_shape
    
    num_row_blocks, num_col_blocks = scale.shape
    
    # Unpack INT4 from INT8
    low_nibble = (x_packed & 0x0F).to(torch.int8)
    high_nibble = ((x_packed >> 4) & 0x0F).to(torch.int8)
    
    # Handle signed INT4 (sign extend from 4-bit)
    if quantized.is_symmetric:
        low_nibble = torch.where(low_nibble > 7, low_nibble - 16, low_nibble)
        high_nibble = torch.where(high_nibble > 7, high_nibble - 16, high_nibble)
    
    # Interleave to restore original order
    x_unpacked = torch.zeros(x_packed.numel() * 2, dtype=torch.float32, device=x_packed.device)
    x_unpacked[0::2] = low_nibble.float()
    x_unpacked[1::2] = high_nibble.float()
    
    # Reshape to blocks
    padded_rows = num_row_blocks * block_size
    padded_cols = num_col_blocks * block_size
    total_elements = padded_rows * padded_cols
    
    x_unpacked = x_unpacked[:total_elements].reshape(
        num_row_blocks, num_col_blocks, block_size, block_size
    )
    
    # Apply scale
    scale_expanded = scale.unsqueeze(-1).unsqueeze(-1)
    x_dequant = x_unpacked * scale_expanded
    
    # Reshape back
    x_dequant = x_dequant.permute(0, 2, 1, 3).reshape(padded_rows, padded_cols)
    
    # Remove padding
    rows, cols = original_shape
    x_dequant = x_dequant[:rows, :cols]
    
    return x_dequant.to(output_dtype)


def get_quantized_size_bytes(
    tensor_shape: tuple[int, ...],
    quant_type: QuantizationType,
    block_size: int = 128,
) -> int:
    """
    Calculate the size in bytes of a quantized tensor.
    
    Args:
        tensor_shape: Shape of the original tensor
        quant_type: Type of quantization
        block_size: Block size for per-block quantization
        
    Returns:
        Size in bytes including data and scales
    """
    numel = math.prod(tensor_shape)
    
    if quant_type == QuantizationType.NONE:
        return numel * 2  # FP16
    
    elif quant_type == QuantizationType.INT8_PER_CHANNEL:
        data_bytes = numel * 1  # INT8
        scale_bytes = tensor_shape[0] * 4  # FP32 scales
        return data_bytes + scale_bytes
    
    elif quant_type == QuantizationType.INT8_PER_TENSOR:
        data_bytes = numel * 1  # INT8
        scale_bytes = 4  # Single FP32 scale
        return data_bytes + scale_bytes
    
    elif quant_type == QuantizationType.INT4_PER_BLOCK:
        data_bytes = math.ceil(numel / 2)  # INT4 packed
        if len(tensor_shape) == 2:
            rows, cols = tensor_shape
            num_blocks = math.ceil(rows / block_size) * math.ceil(cols / block_size)
        else:
            num_blocks = math.ceil(numel / (block_size * block_size))
        scale_bytes = num_blocks * 4  # FP32 scales
        return data_bytes + scale_bytes
    
    elif quant_type == QuantizationType.INT4_PER_CHANNEL:
        data_bytes = math.ceil(numel / 2)  # INT4 packed
        scale_bytes = tensor_shape[0] * 4  # FP32 scales
        return data_bytes + scale_bytes
    
    elif quant_type in (QuantizationType.FP8_E4M3, QuantizationType.FP8_E5M2):
        return numel * 1  # FP8
    
    return numel * 2  # Default to FP16


def compute_compression_ratio(
    original_dtype: torch.dtype,
    quant_type: QuantizationType,
    block_size: int = 128,
) -> float:
    """
    Compute the compression ratio for a quantization type.
    
    Args:
        original_dtype: Original tensor dtype
        quant_type: Target quantization type
        block_size: Block size for per-block quantization
        
    Returns:
        Compression ratio (original_size / compressed_size)
    """
    if original_dtype == torch.float32:
        original_bits = 32
    elif original_dtype in (torch.float16, torch.bfloat16):
        original_bits = 16
    else:
        original_bits = 16  # Assume FP16
    
    if quant_type == QuantizationType.NONE:
        return 1.0
    elif quant_type in (QuantizationType.INT8_PER_CHANNEL, QuantizationType.INT8_PER_TENSOR):
        # 8 bits per value + small scale overhead
        return original_bits / 8.5  # Approximate
    elif quant_type in (QuantizationType.INT4_PER_BLOCK, QuantizationType.INT4_PER_CHANNEL):
        # 4 bits per value + scale overhead
        return original_bits / 4.5  # Approximate
    elif quant_type in (QuantizationType.FP8_E4M3, QuantizationType.FP8_E5M2):
        return original_bits / 8.0
    
    return 1.0
