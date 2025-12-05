"""
KV Split Quantization for Apple Neural Engine

This module implements K8V4 differentiated quantization for KV caches,
where Keys use INT8 (8-bit) and Values use INT4 (4-bit) quantization.

Key Insight:
Keys require higher precision for accurate attention score computation,
while Values can tolerate lower precision as they're only used for
weighted averaging in the attention output.

Memory Savings:
- Standard FP16 KV: 2 bytes/element × 2 (K+V) = 4 bytes
- K8V4: 1 byte (K) + 0.5 byte (V) = 1.5 bytes
- Compression: ~2.67x reduction

References:
- KVSplit: Memory-Efficient KV Cache (https://github.com/yonsei-cysec/KVSplit)
- KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import torch


class KVQuantType(Enum):
    """Quantization types for KV cache."""

    NONE = "none"              # FP16 for both K and V
    K8V8 = "k8v8"              # INT8 for both K and V
    K8V4 = "k8v4"              # INT8 for K, INT4 for V (recommended)
    K4V4 = "k4v4"              # INT4 for both K and V (aggressive)
    K8V2 = "k8v2"              # INT8 for K, INT2 for V (experimental)


@dataclass
class KVSplitConfig:
    """Configuration for KV Split quantization."""

    quant_type: KVQuantType = KVQuantType.K8V4
    # Quantization parameters
    k_block_size: int = 128     # Block size for key quantization
    v_block_size: int = 64      # Block size for value quantization
    symmetric: bool = True      # Use symmetric quantization
    # Residual quantization (for V with very low bits)
    use_residual: bool = False  # Store residual for V
    # Per-head quantization for better accuracy
    per_head: bool = True       # Quantize per attention head


@dataclass
class QuantizedKV:
    """Container for quantized KV data."""

    k_data: torch.Tensor           # Quantized keys (INT8)
    k_scale: torch.Tensor          # Key scales
    v_data: torch.Tensor           # Quantized values (INT4 packed or INT8)
    v_scale: torch.Tensor          # Value scales
    original_shape: tuple[int, ...]  # (batch, heads, seq, head_dim)
    quant_type: KVQuantType        # Quantization type used

    def memory_bytes(self) -> int:
        """Calculate total memory usage in bytes."""
        k_bytes = self.k_data.element_size() * self.k_data.nelement()
        v_bytes = self.v_data.element_size() * self.v_data.nelement()
        scale_bytes = (
            self.k_scale.element_size() * self.k_scale.nelement() +
            self.v_scale.element_size() * self.v_scale.nelement()
        )
        return k_bytes + v_bytes + scale_bytes


def quantize_keys_int8(
    k: torch.Tensor,
    per_head: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize keys to INT8 with per-head or per-tensor scaling.

    Keys require higher precision for accurate attention scores,
    so we use INT8 (8-bit) quantization.

    Args:
        k: Keys tensor (batch, heads, seq_len, head_dim)
        per_head: Use per-head scaling (recommended)

    Returns:
        Tuple of (quantized_keys, scales)
    """
    qmax = 127  # INT8 symmetric range: [-127, 127]

    if per_head:
        # Compute scale per head
        # Shape: (batch, heads, 1, 1)
        abs_max = k.abs().amax(dim=(-2, -1), keepdim=True)
    else:
        # Global scale
        abs_max = k.abs().amax(keepdim=True)

    scale = abs_max / qmax
    scale = torch.clamp(scale, min=1e-10)

    # Quantize
    k_quant = torch.round(k / scale).clamp(-127, 127).to(torch.int8)

    return k_quant, scale.squeeze()


def dequantize_keys_int8(
    k_quant: torch.Tensor,
    scale: torch.Tensor,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Dequantize INT8 keys back to floating point.

    Args:
        k_quant: Quantized keys (INT8)
        scale: Quantization scales
        output_dtype: Output data type

    Returns:
        Dequantized keys
    """
    # Reshape scale for broadcasting
    if scale.ndim < k_quant.ndim:
        # Add dimensions for broadcasting
        while scale.ndim < k_quant.ndim:
            scale = scale.unsqueeze(-1)

    return (k_quant.float() * scale).to(output_dtype)


def quantize_values_int4(
    v: torch.Tensor,
    per_head: bool = True,
    block_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize values to INT4 with per-head or per-block scaling.

    Values can tolerate lower precision as they're only used for
    weighted averaging in attention output.

    INT4 values are packed into INT8 storage (2 values per byte).

    Args:
        v: Values tensor (batch, heads, seq_len, head_dim)
        per_head: Use per-head scaling (recommended)
        block_size: Block size for per-block quantization

    Returns:
        Tuple of (packed_quantized_values, scales)
    """
    qmax = 7  # INT4 symmetric range: [-7, 7]
    batch, heads, seq_len, head_dim = v.shape

    if per_head:
        # Compute scale per head
        abs_max = v.abs().amax(dim=(-2, -1), keepdim=True)
    else:
        # Per-block scaling for finer granularity
        num_blocks = math.ceil(seq_len * head_dim / block_size)
        # Flatten and reshape into blocks
        v_flat = v.reshape(batch, heads, -1)
        padded_size = num_blocks * block_size
        if v_flat.shape[-1] < padded_size:
            v_flat = torch.nn.functional.pad(
                v_flat, (0, padded_size - v_flat.shape[-1])
            )
        v_blocks = v_flat.reshape(batch, heads, num_blocks, block_size)
        abs_max = v_blocks.abs().amax(dim=-1, keepdim=True)

    scale = abs_max / qmax
    scale = torch.clamp(scale, min=1e-10)

    # Quantize to INT4 range
    v_quant = torch.round(v / scale).clamp(-8, 7).to(torch.int8)

    # Pack INT4 into INT8 (2 values per byte)
    v_flat = v_quant.reshape(-1)
    if v_flat.numel() % 2 != 0:
        v_flat = torch.nn.functional.pad(v_flat, (0, 1))

    # Pack: low nibble = even indices, high nibble = odd indices
    v_packed = (v_flat[0::2] & 0x0F) | ((v_flat[1::2] & 0x0F) << 4)
    v_packed = v_packed.to(torch.int8)

    # Reshape packed data
    packed_shape = list(v.shape)
    packed_shape[-1] = packed_shape[-1] // 2 + (1 if head_dim % 2 else 0)
    v_packed = v_packed.reshape(batch, heads, seq_len, -1)

    return v_packed, scale.squeeze()


def dequantize_values_int4(
    v_packed: torch.Tensor,
    scale: torch.Tensor,
    original_shape: tuple[int, ...],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Dequantize packed INT4 values back to floating point.

    Args:
        v_packed: Packed quantized values (INT8 with 2 INT4 per byte)
        scale: Quantization scales
        original_shape: Original tensor shape
        output_dtype: Output data type

    Returns:
        Dequantized values
    """
    batch, heads, seq_len, head_dim = original_shape

    # Unpack INT4 from INT8
    v_flat = v_packed.reshape(-1)
    low_nibble = (v_flat & 0x0F).to(torch.int8)
    high_nibble = ((v_flat >> 4) & 0x0F).to(torch.int8)

    # Sign extend from 4-bit
    low_nibble = torch.where(low_nibble > 7, low_nibble - 16, low_nibble)
    high_nibble = torch.where(high_nibble > 7, high_nibble - 16, high_nibble)

    # Interleave to restore original order
    total_elements = batch * heads * seq_len * head_dim
    v_unpacked = torch.zeros(total_elements, dtype=torch.float32, device=v_packed.device)
    v_unpacked[0::2] = low_nibble[:total_elements // 2 + (1 if total_elements % 2 else 0)].float()
    v_unpacked[1::2] = high_nibble[:total_elements // 2].float()

    # Reshape
    v_unpacked = v_unpacked[:total_elements].reshape(original_shape)

    # Reshape scale for broadcasting
    if scale.ndim < v_unpacked.ndim:
        while scale.ndim < v_unpacked.ndim:
            scale = scale.unsqueeze(-1)

    return (v_unpacked * scale).to(output_dtype)


class KVSplitQuantizer:
    """
    Quantizer for K8V4 differentiated KV cache quantization.

    This class provides methods to quantize and dequantize KV caches
    with different precision for Keys (INT8) and Values (INT4).

    Example:
        config = KVSplitConfig(quant_type=KVQuantType.K8V4)
        quantizer = KVSplitQuantizer(config)

        # Quantize
        quantized = quantizer.quantize(keys, values)

        # Dequantize
        k_dequant, v_dequant = quantizer.dequantize(quantized)
    """

    def __init__(self, config: KVSplitConfig | None = None):
        self.config = config or KVSplitConfig()

    def quantize(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> QuantizedKV:
        """
        Quantize keys and values according to config.

        Args:
            k: Keys tensor (batch, heads, seq_len, head_dim)
            v: Values tensor (batch, heads, seq_len, head_dim)

        Returns:
            QuantizedKV container with quantized data and metadata
        """
        original_shape = k.shape
        quant_type = self.config.quant_type

        if quant_type == KVQuantType.NONE:
            # No quantization
            return QuantizedKV(
                k_data=k.half() if k.dtype != torch.float16 else k,
                k_scale=torch.ones(1, device=k.device),
                v_data=v.half() if v.dtype != torch.float16 else v,
                v_scale=torch.ones(1, device=v.device),
                original_shape=original_shape,
                quant_type=quant_type,
            )

        elif quant_type in (KVQuantType.K8V8, KVQuantType.K8V4, KVQuantType.K8V2):
            # Keys always INT8
            k_quant, k_scale = quantize_keys_int8(k, self.config.per_head)

            if quant_type == KVQuantType.K8V8:
                # Values also INT8
                v_quant, v_scale = quantize_keys_int8(v, self.config.per_head)
            elif quant_type == KVQuantType.K8V4:
                # Values INT4
                v_quant, v_scale = quantize_values_int4(
                    v, self.config.per_head, self.config.v_block_size
                )
            else:  # K8V2
                # Values INT2 (experimental, use INT4 for now)
                v_quant, v_scale = quantize_values_int4(
                    v, self.config.per_head, self.config.v_block_size
                )

            return QuantizedKV(
                k_data=k_quant,
                k_scale=k_scale,
                v_data=v_quant,
                v_scale=v_scale,
                original_shape=original_shape,
                quant_type=quant_type,
            )

        elif quant_type == KVQuantType.K4V4:
            # Both INT4
            k_quant, k_scale = quantize_values_int4(
                k, self.config.per_head, self.config.k_block_size
            )
            v_quant, v_scale = quantize_values_int4(
                v, self.config.per_head, self.config.v_block_size
            )

            return QuantizedKV(
                k_data=k_quant,
                k_scale=k_scale,
                v_data=v_quant,
                v_scale=v_scale,
                original_shape=original_shape,
                quant_type=quant_type,
            )

        raise ValueError(f"Unknown quantization type: {quant_type}")

    def dequantize(
        self,
        quantized: QuantizedKV,
        output_dtype: torch.dtype = torch.float16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Dequantize keys and values back to floating point.

        Args:
            quantized: QuantizedKV container
            output_dtype: Output data type

        Returns:
            Tuple of (dequantized_keys, dequantized_values)
        """
        quant_type = quantized.quant_type
        original_shape = quantized.original_shape

        if quant_type == KVQuantType.NONE:
            return quantized.k_data.to(output_dtype), quantized.v_data.to(output_dtype)

        elif quant_type in (KVQuantType.K8V8, KVQuantType.K8V4, KVQuantType.K8V2):
            # Keys always INT8
            k_dequant = dequantize_keys_int8(
                quantized.k_data, quantized.k_scale, output_dtype
            )

            if quant_type == KVQuantType.K8V8:
                # Values also INT8
                v_dequant = dequantize_keys_int8(
                    quantized.v_data, quantized.v_scale, output_dtype
                )
            else:
                # Values INT4
                v_dequant = dequantize_values_int4(
                    quantized.v_data, quantized.v_scale, original_shape, output_dtype
                )

            return k_dequant, v_dequant

        elif quant_type == KVQuantType.K4V4:
            k_dequant = dequantize_values_int4(
                quantized.k_data, quantized.k_scale, original_shape, output_dtype
            )
            v_dequant = dequantize_values_int4(
                quantized.v_data, quantized.v_scale, original_shape, output_dtype
            )
            return k_dequant, v_dequant

        raise ValueError(f"Unknown quantization type: {quant_type}")

    def compute_compression_ratio(self) -> float:
        """
        Compute the memory compression ratio for the quantization type.

        Returns:
            Compression ratio (original_size / compressed_size)
        """
        quant_type = self.config.quant_type

        # FP16 baseline: 2 bytes per element, 2 tensors (K+V)
        baseline_bytes_per_element = 4.0  # 2 bytes × 2 tensors

        if quant_type == KVQuantType.NONE:
            return 1.0
        elif quant_type == KVQuantType.K8V8:
            # INT8 for both: 1 byte × 2 = 2 bytes
            return baseline_bytes_per_element / 2.0
        elif quant_type == KVQuantType.K8V4:
            # INT8 + INT4 packed: 1 + 0.5 = 1.5 bytes
            return baseline_bytes_per_element / 1.5
        elif quant_type == KVQuantType.K4V4:
            # INT4 × 2 packed: 0.5 + 0.5 = 1 byte
            return baseline_bytes_per_element / 1.0
        elif quant_type == KVQuantType.K8V2:
            # INT8 + INT2: 1 + 0.25 = 1.25 bytes
            return baseline_bytes_per_element / 1.25

        return 1.0


class KVSplitCache:
    """
    KV Cache with K8V4 differentiated quantization.

    This cache stores Keys in INT8 and Values in INT4 format,
    providing ~2.67x memory reduction compared to FP16.

    The cache supports automatic quantization on update and
    dequantization on access, transparent to the user.

    Example:
        config = KVSplitConfig(quant_type=KVQuantType.K8V4)
        cache = KVSplitCache(
            batch_size=1,
            max_seq_len=8192,
            num_heads=32,
            head_dim=128,
            config=config,
        )

        # Update (automatically quantizes)
        k_cached, v_cached = cache.update(keys, values)

        # Access (automatically dequantizes)
        k, v = cache.get_cached_kv()
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        config: KVSplitConfig | None = None,
    ):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.config = config or KVSplitConfig()
        self.quantizer = KVSplitQuantizer(self.config)
        self.current_seq_len = 0

        # Pre-allocate quantized cache
        self._init_cache()

    def _init_cache(self):
        """Initialize pre-allocated quantized cache."""
        quant_type = self.config.quant_type

        # Keys cache (INT8)
        if quant_type in (KVQuantType.K8V8, KVQuantType.K8V4, KVQuantType.K8V2):
            self.k_cache = torch.zeros(
                self.batch_size, self.num_heads, self.max_seq_len, self.head_dim,
                dtype=torch.int8
            )
        elif quant_type == KVQuantType.K4V4:
            # INT4 packed
            packed_dim = self.head_dim // 2 + (1 if self.head_dim % 2 else 0)
            self.k_cache = torch.zeros(
                self.batch_size, self.num_heads, self.max_seq_len, packed_dim,
                dtype=torch.int8
            )
        else:
            self.k_cache = torch.zeros(
                self.batch_size, self.num_heads, self.max_seq_len, self.head_dim,
                dtype=torch.float16
            )

        # Values cache
        if quant_type == KVQuantType.K8V8:
            self.v_cache = torch.zeros(
                self.batch_size, self.num_heads, self.max_seq_len, self.head_dim,
                dtype=torch.int8
            )
        elif quant_type in (KVQuantType.K8V4, KVQuantType.K4V4, KVQuantType.K8V2):
            # INT4 packed
            packed_dim = self.head_dim // 2 + (1 if self.head_dim % 2 else 0)
            self.v_cache = torch.zeros(
                self.batch_size, self.num_heads, self.max_seq_len, packed_dim,
                dtype=torch.int8
            )
        else:
            self.v_cache = torch.zeros(
                self.batch_size, self.num_heads, self.max_seq_len, self.head_dim,
                dtype=torch.float16
            )

        # Scale caches
        self.k_scale_cache = torch.zeros(
            self.batch_size, self.num_heads, self.max_seq_len, 1,
            dtype=torch.float32
        )
        self.v_scale_cache = torch.zeros(
            self.batch_size, self.num_heads, self.max_seq_len, 1,
            dtype=torch.float32
        )

    def update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key-value pairs (quantizing them).

        Args:
            k: New keys (batch, heads, seq_len, head_dim)
            v: New values (batch, heads, seq_len, head_dim)

        Returns:
            Tuple of (dequantized_cached_keys, dequantized_cached_values)
        """
        batch_size, num_heads, new_seq_len, head_dim = k.shape
        start_pos = self.current_seq_len
        end_pos = start_pos + new_seq_len

        if end_pos > self.max_seq_len:
            raise ValueError(
                f"Sequence length {end_pos} exceeds max_seq_len {self.max_seq_len}"
            )

        # Quantize the new K/V
        quantized = self.quantizer.quantize(k, v)

        # Update cache
        self.k_cache[:batch_size, :, start_pos:end_pos, :quantized.k_data.shape[-1]] = quantized.k_data
        self.v_cache[:batch_size, :, start_pos:end_pos, :quantized.v_data.shape[-1]] = quantized.v_data

        # Update scales (per position for now, could optimize)
        k_scale = quantized.k_scale
        v_scale = quantized.v_scale

        # Handle different scale shapes
        if k_scale.ndim == 2:  # (batch, heads)
            k_scale = k_scale.unsqueeze(-1).unsqueeze(-1)
            k_scale = k_scale.expand(-1, -1, new_seq_len, 1)
        if v_scale.ndim == 2:
            v_scale = v_scale.unsqueeze(-1).unsqueeze(-1)
            v_scale = v_scale.expand(-1, -1, new_seq_len, 1)

        self.k_scale_cache[:batch_size, :, start_pos:end_pos, :] = k_scale
        self.v_scale_cache[:batch_size, :, start_pos:end_pos, :] = v_scale

        self.current_seq_len = end_pos

        # Return dequantized values for attention
        return self.get_cached_kv(batch_size)

    def get_cached_kv(
        self,
        batch_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get dequantized cached keys and values up to current position.

        Args:
            batch_size: Batch size (defaults to configured batch_size)

        Returns:
            Tuple of (dequantized_keys, dequantized_values)
        """
        batch_size = batch_size or self.batch_size
        end_pos = self.current_seq_len

        if end_pos == 0:
            return (
                torch.zeros(batch_size, self.num_heads, 0, self.head_dim, dtype=torch.float16),
                torch.zeros(batch_size, self.num_heads, 0, self.head_dim, dtype=torch.float16),
            )

        # Get cached quantized data
        k_cached = self.k_cache[:batch_size, :, :end_pos, :]
        v_cached = self.v_cache[:batch_size, :, :end_pos, :]
        k_scale = self.k_scale_cache[:batch_size, :, :end_pos, :]
        v_scale = self.v_scale_cache[:batch_size, :, :end_pos, :]

        # Create QuantizedKV for dequantization
        original_shape = (batch_size, self.num_heads, end_pos, self.head_dim)

        quantized = QuantizedKV(
            k_data=k_cached,
            k_scale=k_scale,
            v_data=v_cached,
            v_scale=v_scale,
            original_shape=original_shape,
            quant_type=self.config.quant_type,
        )

        return self.quantizer.dequantize(quantized)

    def reset(self):
        """Reset cache for new generation."""
        self.current_seq_len = 0
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.k_scale_cache.zero_()
        self.v_scale_cache.zero_()

    def memory_usage_bytes(self) -> int:
        """Return total memory usage in bytes."""
        return (
            self.k_cache.element_size() * self.k_cache.nelement() +
            self.v_cache.element_size() * self.v_cache.nelement() +
            self.k_scale_cache.element_size() * self.k_scale_cache.nelement() +
            self.v_scale_cache.element_size() * self.v_scale_cache.nelement()
        )

    def compression_ratio(self) -> float:
        """Return the compression ratio compared to FP16."""
        return self.quantizer.compute_compression_ratio()
