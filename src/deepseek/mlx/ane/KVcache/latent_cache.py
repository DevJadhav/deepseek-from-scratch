"""
Unified Memory Latent KV Cache for Apple Silicon

This module provides an enhanced Latent KV Cache that combines:
- Latent compression (14x memory reduction from MLA)
- Unified memory architecture (zero-copy access)
- Optional KVSplit quantization (additional 2.67x reduction)

Total potential memory reduction: 14x × 2.67x ≈ 37x

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                    LATENT KV CACHE                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Standard KV Cache (FP16):                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ K: (batch, heads, seq, head_dim) = 32 × 128 = 4096      │   │
│  │ V: (batch, heads, seq, head_dim) = 32 × 128 = 4096      │   │
│  │ Total: 8192 × 2 bytes = 16KB per token                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  Latent Compression:                                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ C_KV: (batch, seq, d_latent) = 512                      │   │
│  │ Total: 512 × 2 bytes = 1KB per token                     │   │
│  │ Compression: 16x                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  KVSplit Quantization (K8V4):                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ C_KV: (batch, seq, d_latent) as INT8                    │   │
│  │ Total: 512 × 1 byte = 512B per token                     │   │
│  │ Additional compression: 2x                               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Total: 32x compression (16KB → 512B per token)                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

from .unified_kv import (
    ComputeUnit,
    check_unified_memory_available,
    zero_copy_transfer,
)


class LatentQuantType(Enum):
    """Quantization type for latent cache."""

    NONE = "none"          # FP16 latent
    INT8 = "int8"          # INT8 latent (2x compression)
    INT4 = "int4"          # INT4 latent (4x compression)


@dataclass
class LatentCacheConfig:
    """Configuration for Unified Memory Latent KV Cache."""

    batch_size: int = 1
    max_seq_len: int = 8192
    d_latent: int = 512       # Latent dimension (compressed from num_heads × head_dim)
    num_layers: int = 32

    # Quantization
    use_quantization: bool = True
    quant_type: LatentQuantType = LatentQuantType.INT8

    # Memory options
    use_fp16: bool = True
    enable_mmap: bool = False
    mmap_path: str | None = None

    # ANE alignment
    alignment: int = 16


@dataclass
class LatentCacheStats:
    """Statistics for latent cache usage."""

    total_bytes: int = 0
    latent_bytes: int = 0
    scale_bytes: int = 0
    current_seq_len: int = 0
    max_seq_len: int = 0
    num_layers: int = 0
    compression_ratio: float = 1.0
    is_quantized: bool = False

    # Reference comparison
    equivalent_kv_bytes: int = 0

    def __repr__(self) -> str:
        mb = self.total_bytes / (1024 * 1024)
        eq_mb = self.equivalent_kv_bytes / (1024 * 1024)
        return (
            f"LatentCacheStats(total={mb:.2f}MB, "
            f"equivalent_kv={eq_mb:.2f}MB, "
            f"compression={self.compression_ratio:.1f}x, "
            f"seq={self.current_seq_len}/{self.max_seq_len})"
        )


class LatentKVCacheUnified:
    """
    Unified Memory Latent KV Cache with optional quantization.

    This cache stores compressed latent representations instead of
    full K/V tensors, and exploits Apple Silicon's unified memory
    for zero-copy access across compute units.

    Features:
    - Latent compression (~14x memory reduction)
    - Optional INT8/INT4 quantization (additional 2-4x)
    - Zero-copy unified memory access (ANE/GPU/CPU)
    - Per-layer caching for transformer inference
    - Memory-mapped persistence (optional)

    Memory Comparison (32 heads, 128 head_dim, 512 latent):
    - Standard KV (FP16): 2 × 32 × 128 × 2 bytes = 16KB per token
    - Latent (FP16): 512 × 2 bytes = 1KB per token (16x)
    - Latent (INT8): 512 × 1 byte = 512B per token (32x)
    - Latent (INT4): 512 × 0.5 byte = 256B per token (64x)

    Example:
        config = LatentCacheConfig(
            batch_size=1,
            max_seq_len=8192,
            d_latent=512,
            num_layers=32,
            quant_type=LatentQuantType.INT8,
        )
        cache = LatentKVCacheUnified(config)

        # Update layer 0
        latent_cached = cache.update(layer_idx=0, c_kv=latent)

        # Access from any compute unit (zero-copy)
        latent_ane = cache.to_compute_unit(0, ComputeUnit.ANE)
    """

    def __init__(self, config: LatentCacheConfig):
        self.config = config
        self.batch_size = config.batch_size
        self.max_seq_len = config.max_seq_len
        self.d_latent = config.d_latent
        self.num_layers = config.num_layers
        self.use_quantization = config.use_quantization
        self.quant_type = config.quant_type

        self.dtype = torch.float16 if config.use_fp16 else torch.float32
        self.current_seq_len = 0

        # Check unified memory availability
        self.unified_memory_available = check_unified_memory_available()

        # Initialize caches
        self._init_caches()

    def _init_caches(self):
        """Initialize latent caches for all layers."""
        self.latent_caches = []
        self.scale_caches = []

        # Determine storage dtype based on quantization
        if self.use_quantization:
            if self.quant_type == LatentQuantType.INT8:
                storage_dtype = torch.int8
            elif self.quant_type == LatentQuantType.INT4:
                storage_dtype = torch.int8  # Packed INT4
            else:
                storage_dtype = self.dtype
        else:
            storage_dtype = self.dtype

        # Calculate storage dimension
        if self.quant_type == LatentQuantType.INT4 and self.use_quantization:
            # INT4 packed: 2 values per byte
            storage_dim = self.d_latent // 2 + (1 if self.d_latent % 2 else 0)
        else:
            storage_dim = self.d_latent

        for _ in range(self.num_layers):
            # Latent cache: (batch, max_seq_len, storage_dim)
            latent_cache = torch.zeros(
                self.batch_size, self.max_seq_len, storage_dim,
                dtype=storage_dtype
            )

            # Scale cache for quantization
            if self.use_quantization and self.quant_type != LatentQuantType.NONE:
                # Per-position scale for finer granularity
                scale_cache = torch.zeros(
                    self.batch_size, self.max_seq_len, 1,
                    dtype=torch.float32
                )
            else:
                scale_cache = None

            # Move to MPS if available
            if self.unified_memory_available and torch.backends.mps.is_available():
                latent_cache = latent_cache.to("mps")
                if scale_cache is not None:
                    scale_cache = scale_cache.to("mps")

            self.latent_caches.append(latent_cache)
            self.scale_caches.append(scale_cache)

    def _quantize_latent(
        self,
        c_kv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Quantize latent tensor."""
        if not self.use_quantization or self.quant_type == LatentQuantType.NONE:
            return c_kv.to(self.dtype), None

        if self.quant_type == LatentQuantType.INT8:
            # INT8 symmetric quantization
            qmax = 127
            abs_max = c_kv.abs().amax(dim=-1, keepdim=True)
            scale = abs_max / qmax
            scale = torch.clamp(scale, min=1e-10)
            c_kv_quant = torch.round(c_kv / scale).clamp(-127, 127).to(torch.int8)
            return c_kv_quant, scale

        elif self.quant_type == LatentQuantType.INT4:
            # INT4 symmetric quantization with packing
            qmax = 7
            abs_max = c_kv.abs().amax(dim=-1, keepdim=True)
            scale = abs_max / qmax
            scale = torch.clamp(scale, min=1e-10)
            c_kv_quant = torch.round(c_kv / scale).clamp(-8, 7).to(torch.int8)

            # Pack INT4
            c_kv_flat = c_kv_quant.reshape(-1)
            if c_kv_flat.numel() % 2 != 0:
                c_kv_flat = torch.nn.functional.pad(c_kv_flat, (0, 1))
            c_kv_packed = (c_kv_flat[0::2] & 0x0F) | ((c_kv_flat[1::2] & 0x0F) << 4)
            c_kv_packed = c_kv_packed.to(torch.int8)

            # Reshape back
            batch, seq_len, _ = c_kv.shape
            packed_dim = self.d_latent // 2 + (1 if self.d_latent % 2 else 0)
            c_kv_packed = c_kv_packed.reshape(batch, seq_len, packed_dim)

            return c_kv_packed, scale

        return c_kv.to(self.dtype), None

    def _dequantize_latent(
        self,
        c_kv_quant: torch.Tensor,
        scale: torch.Tensor | None,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        """Dequantize latent tensor."""
        if not self.use_quantization or self.quant_type == LatentQuantType.NONE:
            return c_kv_quant.to(self.dtype)

        if scale is None:
            return c_kv_quant.to(self.dtype)

        if self.quant_type == LatentQuantType.INT8:
            return (c_kv_quant.float() * scale).to(self.dtype)

        elif self.quant_type == LatentQuantType.INT4:
            # Unpack INT4
            c_kv_flat = c_kv_quant.reshape(-1)
            low_nibble = (c_kv_flat & 0x0F).to(torch.int8)
            high_nibble = ((c_kv_flat >> 4) & 0x0F).to(torch.int8)

            # Sign extend
            low_nibble = torch.where(low_nibble > 7, low_nibble - 16, low_nibble)
            high_nibble = torch.where(high_nibble > 7, high_nibble - 16, high_nibble)

            # Interleave
            total_elements = batch_size * seq_len * self.d_latent
            c_kv_unpacked = torch.zeros(total_elements, dtype=torch.float32, device=c_kv_quant.device)
            c_kv_unpacked[0::2] = low_nibble[:total_elements // 2 + (1 if total_elements % 2 else 0)].float()
            c_kv_unpacked[1::2] = high_nibble[:total_elements // 2].float()

            c_kv_unpacked = c_kv_unpacked[:total_elements].reshape(batch_size, seq_len, self.d_latent)

            return (c_kv_unpacked * scale).to(self.dtype)

        return c_kv_quant.to(self.dtype)

    def update(
        self,
        layer_idx: int,
        c_kv: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update cache for a specific layer with new compressed latent.

        Args:
            layer_idx: Index of the transformer layer
            c_kv: Compressed KV latent (batch, seq_len, d_latent)

        Returns:
            Full cached latent up to current position (dequantized)
        """
        if layer_idx >= self.num_layers:
            raise ValueError(
                f"Layer index {layer_idx} >= num_layers {self.num_layers}"
            )

        batch_size, new_seq_len, d_latent = c_kv.shape
        start_pos = self.current_seq_len
        end_pos = start_pos + new_seq_len

        if end_pos > self.max_seq_len:
            raise ValueError(
                f"Sequence length {end_pos} exceeds max_seq_len {self.max_seq_len}"
            )

        # Quantize if enabled
        c_kv_quant, scale = self._quantize_latent(c_kv)

        # Ensure tensors are on same device
        latent_cache = self.latent_caches[layer_idx]
        if c_kv_quant.device != latent_cache.device:
            c_kv_quant = c_kv_quant.to(latent_cache.device)
            if scale is not None:
                scale = scale.to(latent_cache.device)

        # Update cache
        latent_cache[:batch_size, start_pos:end_pos, :c_kv_quant.shape[-1]] = c_kv_quant

        if scale is not None and self.scale_caches[layer_idx] is not None:
            self.scale_caches[layer_idx][:batch_size, start_pos:end_pos, :] = scale

        # Update sequence length only on first layer
        if layer_idx == 0:
            self.current_seq_len = end_pos

        # Return dequantized cached latent
        return self.get_cached_latent(layer_idx, batch_size)

    def get_cached_latent(
        self,
        layer_idx: int,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """Get cached latent for a layer up to current position."""
        if layer_idx >= self.num_layers:
            raise ValueError(
                f"Layer index {layer_idx} >= num_layers {self.num_layers}"
            )

        batch_size = batch_size or self.batch_size
        end_pos = self.current_seq_len

        if end_pos == 0:
            return torch.zeros(
                batch_size, 0, self.d_latent,
                dtype=self.dtype, device=self.latent_caches[layer_idx].device
            )

        latent_cached = self.latent_caches[layer_idx][:batch_size, :end_pos, :]
        scale = None
        if self.scale_caches[layer_idx] is not None:
            scale = self.scale_caches[layer_idx][:batch_size, :end_pos, :]

        return self._dequantize_latent(latent_cached, scale, batch_size, end_pos)

    def get_all_cached_latents(
        self,
        batch_size: int | None = None,
    ) -> list[torch.Tensor]:
        """Get cached latents for all layers."""
        return [
            self.get_cached_latent(layer_idx, batch_size)
            for layer_idx in range(self.num_layers)
        ]

    def reset(self):
        """Reset cache for new generation."""
        self.current_seq_len = 0
        for layer_idx in range(self.num_layers):
            self.latent_caches[layer_idx].zero_()
            if self.scale_caches[layer_idx] is not None:
                self.scale_caches[layer_idx].zero_()

    def reset_layer(self, layer_idx: int):
        """Reset cache for a specific layer."""
        if layer_idx >= self.num_layers:
            raise ValueError(
                f"Layer index {layer_idx} >= num_layers {self.num_layers}"
            )
        self.latent_caches[layer_idx].zero_()
        if self.scale_caches[layer_idx] is not None:
            self.scale_caches[layer_idx].zero_()

    def to_compute_unit(
        self,
        layer_idx: int,
        target_unit: ComputeUnit,
    ) -> torch.Tensor:
        """
        Get cache tensor accessible from target compute unit.

        On Apple Silicon with unified memory, this is zero-copy.

        Args:
            layer_idx: Layer index
            target_unit: Target compute unit

        Returns:
            Cached latent accessible from target unit
        """
        latent = self.get_cached_latent(layer_idx)
        return zero_copy_transfer(latent, target_unit)

    def get_stats(self) -> LatentCacheStats:
        """Get memory usage statistics."""
        latent_bytes = sum(
            c.element_size() * c.nelement() for c in self.latent_caches
        )
        scale_bytes = sum(
            c.element_size() * c.nelement()
            for c in self.scale_caches if c is not None
        )

        # Calculate equivalent standard KV cache size
        # Standard: 2 × num_heads × head_dim per token
        # Assume typical: 32 heads × 128 head_dim = 4096 per K, 4096 per V
        typical_kv_dim = 4096 * 2  # K + V
        equivalent_kv_bytes = (
            self.batch_size * self.max_seq_len * typical_kv_dim * 2 *  # FP16
            self.num_layers
        )

        total_bytes = latent_bytes + scale_bytes
        compression_ratio = equivalent_kv_bytes / total_bytes if total_bytes > 0 else 1.0

        return LatentCacheStats(
            total_bytes=total_bytes,
            latent_bytes=latent_bytes,
            scale_bytes=scale_bytes,
            current_seq_len=self.current_seq_len,
            max_seq_len=self.max_seq_len,
            num_layers=self.num_layers,
            compression_ratio=compression_ratio,
            is_quantized=self.use_quantization,
            equivalent_kv_bytes=equivalent_kv_bytes,
        )

    def memory_usage_bytes(self) -> int:
        """Return total memory usage in bytes."""
        return self.get_stats().total_bytes

    @staticmethod
    def compute_latent_memory_reduction(
        num_heads: int,
        head_dim: int,
        d_latent: int,
        quant_type: LatentQuantType = LatentQuantType.NONE,
    ) -> float:
        """
        Compute memory reduction ratio vs standard KV cache.

        Args:
            num_heads: Number of attention heads
            head_dim: Head dimension
            d_latent: Latent dimension
            quant_type: Quantization type

        Returns:
            Memory reduction ratio (e.g., 16.0 means 16x reduction)
        """
        # Standard KV: 2 × num_heads × head_dim × 2 bytes (FP16)
        standard_kv_bytes = 2 * num_heads * head_dim * 2

        # Latent base
        if quant_type == LatentQuantType.NONE:
            latent_bytes = d_latent * 2  # FP16
        elif quant_type == LatentQuantType.INT8:
            latent_bytes = d_latent * 1 + 4  # INT8 + scale
        elif quant_type == LatentQuantType.INT4:
            latent_bytes = d_latent * 0.5 + 4  # INT4 packed + scale
        else:
            latent_bytes = d_latent * 2

        return standard_kv_bytes / latent_bytes if latent_bytes > 0 else 1.0
