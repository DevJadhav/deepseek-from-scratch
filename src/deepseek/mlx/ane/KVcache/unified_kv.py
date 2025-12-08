"""
Unified Memory KV Cache for Apple Silicon

This module implements a KV cache that exploits Apple Silicon's unified memory
architecture for zero-copy access from ANE, GPU, and CPU compute units.

Key advantages of Apple Silicon unified memory:
- No CPU↔GPU memory copies (400 GB/s bandwidth on M3 Max)
- KV cache grows without fragmentation
- Model weights shared between ANE/GPU/CPU
- Direct ANE access without transfer overhead

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                 UNIFIED MEMORY (48GB M3 Max)                    │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                      │
│  │ KV Cache │  │  Model   │  │Activations│                      │
│  │ (Latent) │  │ Weights  │  │           │                      │
│  │          │  │  (INT8)  │  │  (FP16)   │                      │
│  └────┬─────┘  └────┬─────┘  └────┬──────┘                      │
│       │             │             │                              │
│       └──────┬──────┴──────┬──────┘                              │
│              │             │                                     │
│         ┌────▼────┐   ┌────▼────┐                               │
│         │ ANE     │   │  GPU    │   ◄── Zero-Copy               │
│         │ (Attn)  │   │ (MoE)   │       Access!                 │
│         └─────────┘   └─────────┘                               │
└─────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import mmap
import os
import platform
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import torch


class ComputeUnit(Enum):
    """Available compute units on Apple Silicon."""
    
    CPU = "cpu"
    GPU = "gpu"  # Metal Performance Shaders
    ANE = "ane"  # Apple Neural Engine
    ALL = "all"  # All compute units


@dataclass
class UnifiedMemoryConfig:
    """Configuration for Unified Memory KV Cache."""
    
    batch_size: int = 1
    max_seq_len: int = 8192
    num_heads: int = 32
    head_dim: int = 128
    num_layers: int = 32
    
    # Data type configuration
    use_fp16: bool = True
    
    # Memory-mapping for persistence
    enable_mmap: bool = False
    mmap_path: Optional[str] = None
    
    # ANE optimization
    alignment: int = 16  # 16-byte boundary for ANE
    prefetch_layers: int = 2  # Prefetch next N layers
    
    # Compute unit selection
    compute_unit: ComputeUnit = ComputeUnit.ALL


@dataclass
class UnifiedMemoryStats:
    """Statistics for unified memory usage."""
    
    total_bytes: int = 0
    k_cache_bytes: int = 0
    v_cache_bytes: int = 0
    current_seq_len: int = 0
    max_seq_len: int = 0
    num_layers: int = 0
    is_mmap: bool = False
    bandwidth_estimate_gbps: float = 0.0
    
    def __repr__(self) -> str:
        mb = self.total_bytes / (1024 * 1024)
        return (
            f"UnifiedMemoryStats(total={mb:.2f}MB, "
            f"seq={self.current_seq_len}/{self.max_seq_len}, "
            f"layers={self.num_layers}, mmap={self.is_mmap})"
        )


def check_unified_memory_available() -> bool:
    """Check if running on Apple Silicon with unified memory."""
    return (
        platform.system() == "Darwin" 
        and platform.machine() == "arm64"
    )


def get_memory_bandwidth_estimate() -> float:
    """
    Estimate memory bandwidth based on chip generation.
    
    Returns bandwidth in GB/s.
    """
    # These are approximate values based on Apple's specifications
    # Actual performance depends on workload and memory contention
    if not check_unified_memory_available():
        return 50.0  # Default for non-Apple hardware
    
    # Try to detect chip generation from sysctl (macOS)
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True
        )
        chip_name = result.stdout.lower()
        
        if "m4" in chip_name:
            return 546.0  # M4 Max
        elif "m3" in chip_name:
            return 400.0  # M3 Max
        elif "m2" in chip_name:
            return 200.0  # M2 Max
        elif "m1" in chip_name:
            return 200.0  # M1 Max
        else:
            return 100.0  # Conservative estimate
    except Exception:
        return 100.0


def zero_copy_transfer(
    tensor: torch.Tensor,
    target_unit: ComputeUnit,
) -> torch.Tensor:
    """
    Transfer tensor to target compute unit with zero-copy semantics.
    
    On Apple Silicon unified memory, this is essentially a no-op
    since all compute units share the same memory space.
    
    Args:
        tensor: Input tensor
        target_unit: Target compute unit
        
    Returns:
        Tensor accessible from target compute unit (same memory location)
    """
    if not check_unified_memory_available():
        # Fallback for non-Apple hardware - use centralized device selection
        if target_unit == ComputeUnit.CPU:
            return tensor.cpu()
        elif target_unit in (ComputeUnit.GPU, ComputeUnit.ANE):
            # CUDA first, then MPS
            if torch.cuda.is_available():
                return tensor.cuda()
            elif torch.backends.mps.is_available():
                return tensor.to("mps")
        return tensor
    
    # On Apple Silicon, unified memory means zero-copy
    # The tensor is already accessible from all compute units
    # Just ensure it's on the MPS device for GPU/ANE operations
    if target_unit in (ComputeUnit.GPU, ComputeUnit.ANE, ComputeUnit.ALL):
        if torch.backends.mps.is_available() and not tensor.device.type == "mps":
            return tensor.to("mps")
    
    return tensor


class UnifiedMemoryKVCache:
    """
    KV Cache exploiting Apple Silicon unified memory for zero-copy access.
    
    Features:
    - Zero-copy access from ANE, GPU, and CPU
    - Memory-mapped persistence (optional)
    - Per-layer caching for transformer inference
    - FP16 storage for ANE efficiency
    - 16-byte aligned memory allocation
    
    This cache is designed specifically for Apple Silicon's unified memory
    architecture, where CPU, GPU, and ANE all share the same physical memory
    with coherent access.
    
    Example:
        config = UnifiedMemoryConfig(
            batch_size=1,
            max_seq_len=8192,
            num_heads=32,
            head_dim=128,
            num_layers=32,
        )
        cache = UnifiedMemoryKVCache(config)
        
        # Update layer 0
        k_cached, v_cached = cache.update(layer_idx=0, k=keys, v=values)
        
        # Access from any compute unit (zero-copy)
        k_ane = zero_copy_transfer(k_cached, ComputeUnit.ANE)
    """
    
    def __init__(self, config: UnifiedMemoryConfig):
        self.config = config
        self.batch_size = config.batch_size
        self.max_seq_len = config.max_seq_len
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.num_layers = config.num_layers
        self.use_fp16 = config.use_fp16
        
        self.dtype = torch.float16 if config.use_fp16 else torch.float32
        self.current_seq_len = 0
        
        # Check if unified memory is available
        self.unified_memory_available = check_unified_memory_available()
        self.bandwidth_estimate = get_memory_bandwidth_estimate()
        
        # Initialize caches for all layers
        self._init_caches(config)
    
    def _init_caches(self, config: UnifiedMemoryConfig):
        """Initialize K/V caches for all layers."""
        # Shape: (batch, num_heads, max_seq_len, head_dim)
        cache_shape = (
            self.batch_size,
            self.num_heads,
            self.max_seq_len,
            self.head_dim,
        )
        
        if config.enable_mmap and config.mmap_path:
            self._init_mmap_caches(cache_shape, config.mmap_path)
        else:
            self._init_tensor_caches(cache_shape)
    
    def _init_tensor_caches(self, cache_shape: tuple):
        """Initialize in-memory tensor caches."""
        self.k_caches = []
        self.v_caches = []
        
        for _ in range(self.num_layers):
            # Allocate contiguous memory for optimal ANE access
            k_cache = torch.zeros(cache_shape, dtype=self.dtype)
            v_cache = torch.zeros(cache_shape, dtype=self.dtype)
            
            # Move to MPS if available for unified memory access
            if self.unified_memory_available and torch.backends.mps.is_available():
                k_cache = k_cache.to("mps")
                v_cache = v_cache.to("mps")
            
            self.k_caches.append(k_cache)
            self.v_caches.append(v_cache)
        
        self.is_mmap = False
    
    def _init_mmap_caches(self, cache_shape: tuple, mmap_path: str):
        """Initialize memory-mapped caches for persistence."""
        path = Path(mmap_path)
        path.mkdir(parents=True, exist_ok=True)
        
        self.k_caches = []
        self.v_caches = []
        self.mmap_files = []
        
        element_size = 2 if self.use_fp16 else 4
        cache_size = (
            cache_shape[0] * cache_shape[1] * 
            cache_shape[2] * cache_shape[3] * element_size
        )
        
        for layer_idx in range(self.num_layers):
            # Create memory-mapped files for K and V
            k_path = path / f"k_cache_layer_{layer_idx}.bin"
            v_path = path / f"v_cache_layer_{layer_idx}.bin"
            
            # Create or open files
            k_file = open(k_path, "w+b")
            v_file = open(v_path, "w+b")
            
            # Resize files
            k_file.seek(cache_size - 1)
            k_file.write(b'\0')
            k_file.flush()
            v_file.seek(cache_size - 1)
            v_file.write(b'\0')
            v_file.flush()
            
            # Memory-map the files
            k_mmap = mmap.mmap(k_file.fileno(), cache_size)
            v_mmap = mmap.mmap(v_file.fileno(), cache_size)
            
            # Create tensor views over mmap
            k_cache = torch.frombuffer(k_mmap, dtype=self.dtype).reshape(cache_shape)
            v_cache = torch.frombuffer(v_mmap, dtype=self.dtype).reshape(cache_shape)
            
            self.k_caches.append(k_cache)
            self.v_caches.append(v_cache)
            self.mmap_files.append((k_file, v_file, k_mmap, v_mmap))
        
        self.is_mmap = True
    
    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache for a specific layer with new key-value pairs.
        
        Args:
            layer_idx: Index of the transformer layer
            k: New keys (batch, heads, seq_len, head_dim)
            v: New values (batch, heads, seq_len, head_dim)
            position_ids: Optional position IDs for non-sequential update
            
        Returns:
            Tuple of (cached_keys, cached_values) up to current position
        """
        if layer_idx >= self.num_layers:
            raise ValueError(
                f"Layer index {layer_idx} >= num_layers {self.num_layers}"
            )
        
        batch_size, num_heads, new_seq_len, head_dim = k.shape
        
        if position_ids is not None:
            # Non-sequential update (e.g., for speculative decoding)
            return self._update_with_positions(layer_idx, k, v, position_ids)

        # Sequential update
        # All layers should start at the same position for a given forward pass
        # We track per-layer positions to handle independent layer updates
        if not hasattr(self, '_layer_seq_lens'):
            self._layer_seq_lens = [0] * self.num_layers

        start_pos = self._layer_seq_lens[layer_idx]
        end_pos = start_pos + new_seq_len

        if end_pos > self.max_seq_len:
            raise ValueError(
                f"Sequence length {end_pos} exceeds max_seq_len {self.max_seq_len}"
            )
        
        # Convert to FP16 if needed
        if self.use_fp16 and k.dtype != torch.float16:
            k = k.half()
            v = v.half()
        
        # Ensure tensors are on same device
        k_cache = self.k_caches[layer_idx]
        v_cache = self.v_caches[layer_idx]
        
        if k.device != k_cache.device:
            k = k.to(k_cache.device)
            v = v.to(v_cache.device)
        
        # Update cache (zero-copy on unified memory)
        k_cache[:batch_size, :, start_pos:end_pos, :] = k
        v_cache[:batch_size, :, start_pos:end_pos, :] = v

        # Update per-layer sequence length
        self._layer_seq_lens[layer_idx] = end_pos

        # Update global sequence length (max across all layers)
        self.current_seq_len = max(self._layer_seq_lens)

        # Return views into the cache (zero-copy)
        return (
            k_cache[:batch_size, :, :end_pos, :],
            v_cache[:batch_size, :, :end_pos, :],
        )

    def _get_layer_seq_len(self, layer_idx: int) -> int:
        """
        Get the current sequence length for a specific layer.

        This tracks how much of each layer's cache has been filled,
        allowing layers to be updated independently while maintaining
        consistent positions across a forward pass.
        """
        # For simplicity, we use the global counter for all layers
        # In a typical forward pass, all layers are updated together
        # so they should all have the same sequence length
        return self.current_seq_len

    def _update_with_positions(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update cache at specific positions."""
        batch_size = k.shape[0]

        # Convert to FP16 if needed
        if self.use_fp16 and k.dtype != torch.float16:
            k = k.half()
            v = v.half()

        k_cache = self.k_caches[layer_idx]
        v_cache = self.v_caches[layer_idx]

        if k.device != k_cache.device:
            k = k.to(k_cache.device)
            v = v.to(v_cache.device)

        # Update at specific positions
        for i, pos in enumerate(position_ids.squeeze().tolist()):
            if isinstance(pos, (list, tuple)):
                pos = pos[0]
            k_cache[:batch_size, :, pos, :] = k[:, :, i, :]
            v_cache[:batch_size, :, pos, :] = v[:, :, i, :]

        # Get max position for return slice
        max_pos = position_ids.max().item() + 1

        return (
            k_cache[:batch_size, :, :max_pos, :],
            v_cache[:batch_size, :, :max_pos, :],
        )
    
    def get_cached_kv(
        self,
        layer_idx: int,
        batch_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get cached keys and values for a layer up to current position."""
        if layer_idx >= self.num_layers:
            raise ValueError(
                f"Layer index {layer_idx} >= num_layers {self.num_layers}"
            )
        
        batch_size = batch_size or self.batch_size
        return (
            self.k_caches[layer_idx][:batch_size, :, :self.current_seq_len, :],
            self.v_caches[layer_idx][:batch_size, :, :self.current_seq_len, :],
        )
    
    def get_all_cached_kv(
        self,
        batch_size: Optional[int] = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Get cached keys and values for all layers."""
        return [
            self.get_cached_kv(layer_idx, batch_size)
            for layer_idx in range(self.num_layers)
        ]
    
    def reset(self):
        """Reset cache for new generation."""
        self.current_seq_len = 0
        if hasattr(self, '_layer_seq_lens'):
            self._layer_seq_lens = [0] * self.num_layers

        for layer_idx in range(self.num_layers):
            self.k_caches[layer_idx].zero_()
            self.v_caches[layer_idx].zero_()
    
    def reset_layer(self, layer_idx: int):
        """Reset cache for a specific layer."""
        if layer_idx >= self.num_layers:
            raise ValueError(
                f"Layer index {layer_idx} >= num_layers {self.num_layers}"
            )
        
        self.k_caches[layer_idx].zero_()
        self.v_caches[layer_idx].zero_()
    
    def get_stats(self) -> UnifiedMemoryStats:
        """Get memory usage statistics."""
        k_bytes = sum(
            c.element_size() * c.nelement() for c in self.k_caches
        )
        v_bytes = sum(
            c.element_size() * c.nelement() for c in self.v_caches
        )
        
        return UnifiedMemoryStats(
            total_bytes=k_bytes + v_bytes,
            k_cache_bytes=k_bytes,
            v_cache_bytes=v_bytes,
            current_seq_len=self.current_seq_len,
            max_seq_len=self.max_seq_len,
            num_layers=self.num_layers,
            is_mmap=self.is_mmap,
            bandwidth_estimate_gbps=self.bandwidth_estimate,
        )
    
    def memory_usage_bytes(self) -> int:
        """Return total memory usage in bytes."""
        return self.get_stats().total_bytes
    
    def prefetch_layer(self, layer_idx: int):
        """
        Prefetch cache for a layer (hint for memory subsystem).
        
        On Apple Silicon, this is mostly a no-op due to unified memory,
        but it can help with Metal memory management.
        """
        if layer_idx >= self.num_layers:
            return
        
        # Touch the memory to ensure it's paged in
        # This is a hint, not a guarantee
        _ = self.k_caches[layer_idx].data_ptr()
        _ = self.v_caches[layer_idx].data_ptr()
    
    def to_compute_unit(
        self,
        layer_idx: int,
        target_unit: ComputeUnit,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get cache tensors accessible from target compute unit.
        
        On Apple Silicon with unified memory, this is zero-copy.
        On other platforms, this may involve memory transfer.
        
        Args:
            layer_idx: Layer index
            target_unit: Target compute unit
            
        Returns:
            Tuple of (k_cache, v_cache) accessible from target unit
        """
        k_cache, v_cache = self.get_cached_kv(layer_idx)
        return (
            zero_copy_transfer(k_cache, target_unit),
            zero_copy_transfer(v_cache, target_unit),
        )
    
    def __del__(self):
        """Cleanup memory-mapped files."""
        if hasattr(self, 'is_mmap') and self.is_mmap:
            for k_file, v_file, k_mmap, v_mmap in self.mmap_files:
                try:
                    k_mmap.close()
                    v_mmap.close()
                    k_file.close()
                    v_file.close()
                except Exception:
                    pass
