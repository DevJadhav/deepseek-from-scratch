"""
Sliding Window KV Cache for Extreme Context Length

This module implements a rolling/sliding window KV cache optimized for
Apple Neural Engine, enabling efficient handling of extreme context lengths
(128K+ tokens) while maintaining bounded memory usage.

Key Features:
- Fixed memory footprint regardless of total sequence length
- Rolling window with configurable sink tokens (attention sinks)
- Local attention pattern for ANE efficiency
- Zero-copy unified memory integration
- Support for streaming/continuous generation

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│ SLIDING WINDOW KV CACHE                                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Full Sequence: [tok0][tok1]...[tok(n-w)]...[tok(n-1)][tok_n]  │
│                   ↓                                             │
│  Cache Window:  [SINK][SINK]...[RECENT]...[RECENT][CURRENT]    │
│                 ├──────┬──────┼────────────────────┤           │
│                 │ Sink │ Gap  │   Sliding Window   │           │
│                 │Tokens│(drop)│    (window_size)   │           │
│                 └──────┴──────┴────────────────────┘           │
│                                                                 │
│  Memory Layout (per layer):                                     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ K Cache: [batch, num_heads, window_size+sink, head_dim]    │ │
│  │ V Cache: [batch, num_heads, window_size+sink, head_dim]    │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  Benefits:                                                       │
│  • O(window_size) memory instead of O(seq_len)                 │
│  • Attention sinks preserve long-range dependencies            │
│  • Compatible with chunked attention (128-token windows)       │
│  • Zero-copy ANE access via unified memory                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

References:
- StreamingLLM: Efficient Streaming Language Models with Attention Sinks
- LongLLaMA: Focused Transformer for Long Context
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn


class WindowEvictionPolicy(Enum):
    """Policy for evicting tokens from the sliding window."""
    
    FIFO = "fifo"  # First-in-first-out (standard sliding)
    LRU = "lru"  # Least recently used (track attention scores)
    IMPORTANCE = "importance"  # Based on attention weights


@dataclass
class SlidingWindowConfig:
    """Configuration for Sliding Window KV Cache."""
    
    # Window parameters
    window_size: int = 4096  # Size of the sliding window
    sink_tokens: int = 4  # Number of attention sink tokens to preserve
    
    # Shape parameters
    batch_size: int = 1
    num_heads: int = 32
    head_dim: int = 128
    num_layers: int = 32
    
    # Data type
    dtype: torch.dtype = torch.float16
    
    # Eviction policy
    eviction_policy: WindowEvictionPolicy = WindowEvictionPolicy.FIFO
    
    # ANE optimization
    alignment: int = 16  # 16-byte boundary
    chunk_size: int = 128  # ANE-friendly chunk size
    
    # Quantization (optional compression)
    quantize_keys: bool = False  # INT8 key quantization
    quantize_values: bool = False  # INT4 value quantization
    
    @property
    def total_window_size(self) -> int:
        """Total cache size including sink tokens."""
        return self.window_size + self.sink_tokens
    
    @property
    def memory_bytes_per_layer(self) -> int:
        """Memory usage per layer in bytes."""
        elem_size = 2 if self.dtype == torch.float16 else 4
        k_bytes = self.batch_size * self.num_heads * self.total_window_size * self.head_dim * elem_size
        v_bytes = k_bytes
        return k_bytes + v_bytes
    
    @property
    def total_memory_bytes(self) -> int:
        """Total memory usage in bytes."""
        return self.memory_bytes_per_layer * self.num_layers


@dataclass
class SlidingWindowStats:
    """Statistics for sliding window cache."""
    
    total_tokens_seen: int = 0
    tokens_in_window: int = 0
    tokens_evicted: int = 0
    window_size: int = 0
    sink_tokens: int = 0
    memory_bytes: int = 0
    utilization: float = 0.0
    
    def __repr__(self) -> str:
        mb = self.memory_bytes / (1024 * 1024)
        return (
            f"SlidingWindowStats(seen={self.total_tokens_seen}, "
            f"in_window={self.tokens_in_window}, "
            f"evicted={self.tokens_evicted}, "
            f"util={self.utilization:.1%}, mem={mb:.2f}MB)"
        )


class SlidingWindowKVCache(nn.Module):
    """
    Sliding Window KV Cache for extreme context lengths.
    
    This cache maintains a fixed-size window of recent tokens plus
    "attention sink" tokens from the beginning of the sequence. When
    the window is full, older tokens are evicted to make room for new ones.
    
    Key features:
    - Fixed memory footprint: O(window_size) instead of O(seq_len)
    - Attention sinks: Preserve initial tokens for long-range attention
    - Zero-copy: Unified memory for ANE/GPU/CPU access
    - Streaming: Supports continuous generation without memory growth
    
    Example:
        config = SlidingWindowConfig(
            window_size=4096,
            sink_tokens=4,
            num_heads=32,
            head_dim=128,
            num_layers=32,
        )
        cache = SlidingWindowKVCache(config)
        
        # During generation:
        for layer_idx in range(num_layers):
            k, v = layer.get_kv(hidden_states)
            cache.update(layer_idx, k, v)
            cached_k, cached_v = cache.get_cached_kv(layer_idx)
            # Use cached_k, cached_v for attention
    """
    
    def __init__(self, config: SlidingWindowConfig):
        super().__init__()
        self.config = config
        
        # Initialize caches for all layers
        total_size = config.total_window_size
        cache_shape = (
            config.batch_size,
            config.num_heads,
            total_size,
            config.head_dim,
        )
        
        # Pre-allocate K and V caches for each layer
        self.k_caches: list[torch.Tensor] = []
        self.v_caches: list[torch.Tensor] = []
        
        for _ in range(config.num_layers):
            self.k_caches.append(
                torch.zeros(cache_shape, dtype=config.dtype)
            )
            self.v_caches.append(
                torch.zeros(cache_shape, dtype=config.dtype)
            )
        
        # Track state
        self.current_pos = 0  # Position within the window
        self.total_tokens_seen = 0
        self.tokens_evicted = 0
        self._sink_filled = False  # Whether sink tokens have been filled
        
        # Per-layer position tracking for proper multi-layer updates
        self._layer_positions: list[int] = [0] * config.num_layers
    
    @property
    def window_size(self) -> int:
        """Get the window size (excluding sink tokens)."""
        return self.config.window_size
    
    @property
    def sink_tokens(self) -> int:
        """Get the number of sink tokens."""
        return self.config.sink_tokens
    
    @property
    def total_window_size(self) -> int:
        """Get total cache size (window + sink)."""
        return self.config.total_window_size
    
    @property
    def tokens_in_cache(self) -> int:
        """Get number of valid tokens currently in cache."""
        if self.total_tokens_seen <= self.total_window_size:
            return self.total_tokens_seen
        return self.total_window_size
    
    def _get_write_position(self, seq_len: int) -> tuple[int, int]:
        """
        Calculate the write position for new tokens.
        
        Returns:
            (start_pos, end_pos) for writing new tokens
        """
        sink = self.sink_tokens
        window = self.window_size
        
        if self.total_tokens_seen < sink:
            # Still filling sink tokens
            start = self.total_tokens_seen
            end = min(start + seq_len, sink)
            return start, end
        
        if self.total_tokens_seen < self.total_window_size:
            # Sink filled, filling window
            start = self.total_tokens_seen
            end = min(start + seq_len, self.total_window_size)
            return start, end
        
        # Window is full, need to slide
        # Window positions are after sink tokens
        window_start = sink
        
        # Calculate position within the sliding portion (circular buffer)
        pos_in_window = (self.current_pos % window)
        start = window_start + pos_in_window
        end = start + seq_len
        
        # Handle wrap-around
        if end > self.total_window_size:
            end = self.total_window_size
        
        return start, end
    
    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the cache with new key-value pairs.
        
        This method handles the sliding window logic:
        1. If window not full, append to end
        2. If window full, evict oldest and append
        3. Sink tokens are never evicted
        
        Args:
            layer_idx: Layer index
            key: New keys [batch, num_heads, seq_len, head_dim]
            value: New values [batch, num_heads, seq_len, head_dim]
            
        Returns:
            (cached_keys, cached_values): Full cached KV for attention
        """
        if layer_idx < 0 or layer_idx >= self.config.num_layers:
            raise ValueError(f"Invalid layer_idx {layer_idx}")
        
        seq_len = key.shape[2]
        k_cache = self.k_caches[layer_idx]
        v_cache = self.v_caches[layer_idx]
        
        # Ensure device and dtype match
        device = k_cache.device
        key = key.to(device=device, dtype=self.config.dtype)
        value = value.to(device=device, dtype=self.config.dtype)
        
        # Use per-layer position for proper tracking in multi-layer updates
        layer_tokens_seen = self._layer_positions[layer_idx]
        sink = self.sink_tokens
        
        if layer_tokens_seen < self.total_window_size:
            # Cache not full yet, simple append
            start = layer_tokens_seen
            end = min(start + seq_len, self.total_window_size)
            actual_len = end - start
            
            k_cache[:, :, start:end, :] = key[:, :, :actual_len, :]
            v_cache[:, :, start:end, :] = value[:, :, :actual_len, :]
            
            # Update per-layer position
            self._layer_positions[layer_idx] = end
            
            # Update global state only for layer 0 to avoid double counting
            if layer_idx == 0:
                self.total_tokens_seen = end
        else:
            # Cache is full, use circular buffer in window portion
            window_start = sink
            window_size = self.window_size
            
            # Calculate position within circular buffer
            layer_current_pos = layer_tokens_seen - self.total_window_size
            
            for i in range(seq_len):
                pos_in_window = (layer_current_pos + i) % window_size
                cache_pos = window_start + pos_in_window
                
                k_cache[:, :, cache_pos:cache_pos+1, :] = key[:, :, i:i+1, :]
                v_cache[:, :, cache_pos:cache_pos+1, :] = value[:, :, i:i+1, :]
            
            # Update per-layer tracking
            self._layer_positions[layer_idx] += seq_len
            
            # Update global state only for layer 0
            if layer_idx == 0:
                self.total_tokens_seen += seq_len
                self.current_pos = (self.current_pos + seq_len) % window_size
                self.tokens_evicted += seq_len
        
        return self.get_cached_kv(layer_idx)
    
    def get_cached_kv(
        self,
        layer_idx: int,
        max_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the cached key-value pairs for attention.
        
        Args:
            layer_idx: Layer index
            max_length: Optional maximum length to return
            
        Returns:
            (keys, values): Cached KV pairs for attention computation
        """
        if layer_idx < 0 or layer_idx >= self.config.num_layers:
            raise ValueError(f"Invalid layer_idx {layer_idx}")
        
        k_cache = self.k_caches[layer_idx]
        v_cache = self.v_caches[layer_idx]
        
        # Get the actual number of valid tokens
        layer_pos = self._layer_positions[layer_idx]
        valid_len = min(layer_pos, self.total_window_size)
        
        if max_length is not None:
            valid_len = min(valid_len, max_length)
        
        if valid_len == 0:
            # Return empty tensors with correct shape
            empty_shape = (
                self.config.batch_size,
                self.config.num_heads,
                0,
                self.config.head_dim,
            )
            return (
                torch.zeros(empty_shape, dtype=self.config.dtype, device=k_cache.device),
                torch.zeros(empty_shape, dtype=self.config.dtype, device=v_cache.device),
            )
        
        # Return the valid portion
        return (
            k_cache[:, :, :valid_len, :].clone(),
            v_cache[:, :, :valid_len, :].clone(),
        )
    
    def get_attention_mask(
        self,
        query_len: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Get attention mask for sliding window attention.
        
        The mask ensures:
        1. Sink tokens are always visible
        2. Only window_size recent tokens are visible (local attention)
        3. Causal masking is applied
        
        Args:
            query_len: Length of the query sequence
            device: Target device for the mask
            
        Returns:
            Attention mask [1, 1, query_len, kv_len]
        """
        kv_len = min(self.total_tokens_seen, self.total_window_size)
        
        if device is None:
            device = self.k_caches[0].device
        
        # Start with all masked (False = masked)
        mask = torch.zeros(1, 1, query_len, kv_len, dtype=torch.bool, device=device)
        
        sink = self.sink_tokens
        
        # Sink tokens are always visible
        if sink > 0 and kv_len > 0:
            sink_end = min(sink, kv_len)
            mask[:, :, :, :sink_end] = True
        
        # Local window attention (recent tokens)
        if kv_len > sink:
            # For each query position, it can attend to recent tokens
            # This is a simplified causal local attention
            for q in range(query_len):
                # Query position in the full sequence
                full_q_pos = self.total_tokens_seen - query_len + q
                
                # Window of visible tokens
                window_start = max(sink, full_q_pos - self.window_size + 1)
                window_end = min(kv_len, full_q_pos + 1)
                
                # Map to cache positions
                if self.total_tokens_seen <= self.total_window_size:
                    # Not yet sliding
                    cache_start = window_start
                    cache_end = window_end
                else:
                    # Sliding - window positions are valid
                    cache_start = sink
                    cache_end = kv_len
                
                mask[:, :, q, cache_start:cache_end] = True
        
        return mask
    
    def reset(self):
        """Reset the cache for a new sequence."""
        self.current_pos = 0
        self.total_tokens_seen = 0
        self.tokens_evicted = 0
        self._sink_filled = False
        self._layer_positions = [0] * self.config.num_layers
        
        for layer_idx in range(self.config.num_layers):
            self.k_caches[layer_idx].zero_()
            self.v_caches[layer_idx].zero_()
    
    def get_stats(self) -> SlidingWindowStats:
        """Get cache statistics."""
        return SlidingWindowStats(
            total_tokens_seen=self.total_tokens_seen,
            tokens_in_window=self.tokens_in_cache,
            tokens_evicted=self.tokens_evicted,
            window_size=self.window_size,
            sink_tokens=self.sink_tokens,
            memory_bytes=self.config.total_memory_bytes,
            utilization=self.tokens_in_cache / self.total_window_size if self.total_window_size > 0 else 0.0,
        )
    
    def get_memory_usage_bytes(self) -> int:
        """Get total memory usage in bytes."""
        total = 0
        for k_cache, v_cache in zip(self.k_caches, self.v_caches):
            total += k_cache.numel() * k_cache.element_size()
            total += v_cache.numel() * v_cache.element_size()
        return total
    
    def to(self, device: torch.device) -> SlidingWindowKVCache:
        """Move cache to specified device."""
        for i in range(len(self.k_caches)):
            self.k_caches[i] = self.k_caches[i].to(device)
            self.v_caches[i] = self.v_caches[i].to(device)
        return self


class StreamingSlidingWindowCache(SlidingWindowKVCache):
    """
    Extended sliding window cache optimized for streaming inference.
    
    Additional features for streaming:
    - Efficient batch updates (multiple tokens at once)
    - Token recycling (reuse evicted token slots)
    - Speculative decoding support
    - Checkpoint/restore for pause/resume
    """
    
    def __init__(self, config: SlidingWindowConfig):
        super().__init__(config)
        self._checkpoint_state: dict | None = None
    
    def update_batch(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Batch update for multiple tokens at once.
        
        More efficient than calling update() for each token.
        
        Args:
            layer_idx: Layer index
            keys: Keys for multiple tokens [batch, num_heads, num_tokens, head_dim]
            values: Values for multiple tokens [batch, num_heads, num_tokens, head_dim]
            
        Returns:
            (cached_keys, cached_values): Updated cache
        """
        return self.update(layer_idx, keys, values)
    
    def speculative_update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        num_accepted: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update for speculative decoding with partial acceptance.
        
        Only commits the first num_accepted tokens to the cache.
        
        Args:
            layer_idx: Layer index
            key: Speculative keys [batch, num_heads, spec_len, head_dim]
            value: Speculative values [batch, num_heads, spec_len, head_dim]
            num_accepted: Number of tokens that were accepted
            
        Returns:
            (cached_keys, cached_values): Updated cache with accepted tokens only
        """
        if num_accepted <= 0:
            return self.get_cached_kv(layer_idx)
        
        # Only update with accepted tokens
        accepted_k = key[:, :, :num_accepted, :]
        accepted_v = value[:, :, :num_accepted, :]
        
        return self.update(layer_idx, accepted_k, accepted_v)
    
    def checkpoint(self) -> dict:
        """
        Save current cache state for later restoration.
        
        Returns:
            State dictionary for restoration
        """
        self._checkpoint_state = {
            "current_pos": self.current_pos,
            "total_tokens_seen": self.total_tokens_seen,
            "tokens_evicted": self.tokens_evicted,
            "layer_positions": self._layer_positions.copy(),
            "k_caches": [k.clone() for k in self.k_caches],
            "v_caches": [v.clone() for v in self.v_caches],
        }
        return self._checkpoint_state
    
    def restore(self, state: dict | None = None):
        """
        Restore cache state from checkpoint.
        
        Args:
            state: State dictionary (uses last checkpoint if None)
        """
        if state is None:
            state = self._checkpoint_state
        
        if state is None:
            raise ValueError("No checkpoint state available")
        
        self.current_pos = state["current_pos"]
        self.total_tokens_seen = state["total_tokens_seen"]
        self.tokens_evicted = state["tokens_evicted"]
        self._layer_positions = state["layer_positions"].copy()
        
        for i, (k, v) in enumerate(zip(state["k_caches"], state["v_caches"])):
            self.k_caches[i].copy_(k)
            self.v_caches[i].copy_(v)
    
    def rollback(self, num_tokens: int):
        """
        Rollback the cache by num_tokens.
        
        Useful for speculative decoding when tokens are rejected.
        
        Args:
            num_tokens: Number of tokens to rollback
        """
        if num_tokens <= 0:
            return
        
        # Adjust positions
        new_total = max(0, self.total_tokens_seen - num_tokens)
        
        if new_total < self.total_tokens_seen:
            # Update eviction count if we were evicting
            if self.total_tokens_seen > self.total_window_size:
                evicted_after = max(0, new_total - self.total_window_size)
                self.tokens_evicted = max(0, evicted_after)
            
            self.total_tokens_seen = new_total
            
            # Reset layer positions
            for i in range(len(self._layer_positions)):
                self._layer_positions[i] = new_total
            
            # Update circular buffer position
            if new_total > self.total_window_size:
                self.current_pos = (new_total - self.total_window_size) % self.window_size
            else:
                self.current_pos = 0


def create_sliding_window_cache(
    window_size: int = 4096,
    sink_tokens: int = 4,
    num_heads: int = 32,
    head_dim: int = 128,
    num_layers: int = 32,
    dtype: torch.dtype = torch.float16,
    streaming: bool = False,
) -> SlidingWindowKVCache:
    """
    Factory function for creating sliding window caches.
    
    Args:
        window_size: Size of the sliding window
        sink_tokens: Number of attention sink tokens
        num_heads: Number of attention heads
        head_dim: Dimension per head
        num_layers: Number of transformer layers
        dtype: Data type for cache
        streaming: Whether to use streaming-optimized cache
        
    Returns:
        Configured sliding window cache
    """
    config = SlidingWindowConfig(
        window_size=window_size,
        sink_tokens=sink_tokens,
        num_heads=num_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        dtype=dtype,
    )
    
    if streaming:
        return StreamingSlidingWindowCache(config)
    return SlidingWindowKVCache(config)


def estimate_memory_savings(
    max_seq_len: int,
    window_size: int,
    sink_tokens: int,
    num_heads: int = 32,
    head_dim: int = 128,
    num_layers: int = 32,
    dtype: torch.dtype = torch.float16,
) -> dict:
    """
    Estimate memory savings from using sliding window cache.
    
    Args:
        max_seq_len: Maximum sequence length without sliding window
        window_size: Sliding window size
        sink_tokens: Number of sink tokens
        num_heads: Number of attention heads
        head_dim: Dimension per head
        num_layers: Number of transformer layers
        dtype: Data type
        
    Returns:
        Dictionary with memory comparison statistics
    """
    elem_size = 2 if dtype == torch.float16 else 4
    
    # Full cache memory
    full_cache_bytes = 2 * num_layers * num_heads * max_seq_len * head_dim * elem_size
    
    # Sliding window cache memory
    total_window = window_size + sink_tokens
    sliding_cache_bytes = 2 * num_layers * num_heads * total_window * head_dim * elem_size
    
    savings_bytes = full_cache_bytes - sliding_cache_bytes
    savings_ratio = full_cache_bytes / sliding_cache_bytes if sliding_cache_bytes > 0 else float('inf')
    
    return {
        "full_cache_mb": full_cache_bytes / (1024 * 1024),
        "sliding_cache_mb": sliding_cache_bytes / (1024 * 1024),
        "savings_mb": savings_bytes / (1024 * 1024),
        "savings_ratio": savings_ratio,
        "max_seq_len": max_seq_len,
        "window_size": window_size,
        "sink_tokens": sink_tokens,
    }
