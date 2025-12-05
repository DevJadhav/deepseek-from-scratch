"""
Tests for ANE KV Cache Implementations

Tests for ANEKVCache, ANELatentKVCache, and ANESlidingWindowCache.
"""

import pytest
import torch

from deepseek.mlx.ane.attention.kv_cache import (
    ANEKVCache,
    ANEKVCacheConfig,
    ANELatentKVCache,
    ANESlidingWindowCache,
)


class TestANEKVCache:
    """Tests for standard ANE KV Cache."""

    def test_initialization(self):
        """Test cache initialization."""
        cache = ANEKVCache(
            batch_size=2,
            max_seq_len=1024,
            num_heads=8,
            head_dim=64,
            use_fp16=True,
        )

        assert cache.k_cache.shape == (2, 8, 1024, 64)
        assert cache.v_cache.shape == (2, 8, 1024, 64)
        assert cache.k_cache.dtype == torch.float16
        assert cache.current_seq_len == 0

    def test_update(self):
        """Test cache update operation."""
        cache = ANEKVCache(
            batch_size=2,
            max_seq_len=1024,
            num_heads=8,
            head_dim=64,
        )

        # Add first batch of tokens
        k = torch.randn(2, 8, 32, 64)
        v = torch.randn(2, 8, 32, 64)

        k_out, v_out = cache.update(k, v)

        assert k_out.shape == (2, 8, 32, 64)
        assert v_out.shape == (2, 8, 32, 64)
        assert cache.current_seq_len == 32

    def test_incremental_update(self):
        """Test incremental cache updates."""
        cache = ANEKVCache(
            batch_size=1,
            max_seq_len=128,
            num_heads=4,
            head_dim=32,
        )

        # First update
        k1 = torch.randn(1, 4, 16, 32)
        v1 = torch.randn(1, 4, 16, 32)
        cache.update(k1, v1)
        assert cache.current_seq_len == 16

        # Second update
        k2 = torch.randn(1, 4, 8, 32)
        v2 = torch.randn(1, 4, 8, 32)
        k_out, v_out = cache.update(k2, v2)

        assert cache.current_seq_len == 24
        assert k_out.shape == (1, 4, 24, 32)

    def test_get_cached_kv(self):
        """Test retrieving cached KV."""
        cache = ANEKVCache(
            batch_size=2,
            max_seq_len=128,
            num_heads=4,
            head_dim=32,
        )

        k = torch.randn(2, 4, 16, 32)
        v = torch.randn(2, 4, 16, 32)
        cache.update(k, v)

        k_cached, v_cached = cache.get_cached_kv()
        assert k_cached.shape == (2, 4, 16, 32)

    def test_reset(self):
        """Test cache reset."""
        cache = ANEKVCache(
            batch_size=1,
            max_seq_len=64,
            num_heads=4,
            head_dim=32,
        )

        k = torch.randn(1, 4, 16, 32)
        v = torch.randn(1, 4, 16, 32)
        cache.update(k, v)

        cache.reset()
        assert cache.current_seq_len == 0

    def test_exceeds_max_seq_len(self):
        """Test error when exceeding max sequence length."""
        cache = ANEKVCache(
            batch_size=1,
            max_seq_len=32,
            num_heads=4,
            head_dim=32,
        )

        k = torch.randn(1, 4, 64, 32)
        v = torch.randn(1, 4, 64, 32)

        with pytest.raises(ValueError, match="exceeds max_seq_len"):
            cache.update(k, v)

    def test_fp16_conversion(self):
        """Test automatic FP16 conversion."""
        cache = ANEKVCache(
            batch_size=1,
            max_seq_len=64,
            num_heads=4,
            head_dim=32,
            use_fp16=True,
        )

        k = torch.randn(1, 4, 16, 32, dtype=torch.float32)
        v = torch.randn(1, 4, 16, 32, dtype=torch.float32)

        k_out, v_out = cache.update(k, v)
        assert k_out.dtype == torch.float16

    def test_memory_usage(self):
        """Test memory usage calculation."""
        cache = ANEKVCache(
            batch_size=1,
            max_seq_len=1024,
            num_heads=8,
            head_dim=64,
            use_fp16=True,
        )

        # Each tensor: 1 * 8 * 1024 * 64 * 2 bytes (FP16)
        expected_per_cache = 1 * 8 * 1024 * 64 * 2
        expected_total = expected_per_cache * 2  # K and V

        assert cache.memory_usage_bytes() == expected_total


class TestANELatentKVCache:
    """Tests for latent KV cache."""

    def test_initialization(self):
        """Test latent cache initialization."""
        cache = ANELatentKVCache(
            batch_size=2,
            max_seq_len=1024,
            d_latent=512,
            use_fp16=True,
        )

        assert cache.latent_cache.shape == (2, 1024, 512)
        assert cache.latent_cache.dtype == torch.float16
        assert cache.current_seq_len == 0

    def test_update(self):
        """Test latent cache update."""
        cache = ANELatentKVCache(
            batch_size=2,
            max_seq_len=1024,
            d_latent=512,
        )

        c_kv = torch.randn(2, 32, 512)
        c_out = cache.update(c_kv)

        assert c_out.shape == (2, 32, 512)
        assert cache.current_seq_len == 32

    def test_incremental_update(self):
        """Test incremental latent updates."""
        cache = ANELatentKVCache(
            batch_size=1,
            max_seq_len=128,
            d_latent=256,
        )

        # First update
        c1 = torch.randn(1, 16, 256)
        cache.update(c1)

        # Second update
        c2 = torch.randn(1, 8, 256)
        c_out = cache.update(c2)

        assert cache.current_seq_len == 24
        assert c_out.shape == (1, 24, 256)

    def test_get_cached_latent(self):
        """Test retrieving cached latent."""
        cache = ANELatentKVCache(
            batch_size=2,
            max_seq_len=128,
            d_latent=256,
        )

        c_kv = torch.randn(2, 16, 256)
        cache.update(c_kv)

        cached = cache.get_cached_latent()
        assert cached.shape == (2, 16, 256)

    def test_memory_reduction_computation(self):
        """Test memory reduction ratio calculation."""
        # 32 heads, 128 head_dim, 512 latent
        # Standard: 2 * 32 * 128 = 8192
        # Latent: 512
        # Reduction: 16x
        ratio = ANELatentKVCache.compute_memory_reduction(
            num_heads=32,
            head_dim=128,
            d_latent=512,
        )
        assert ratio == 16.0

    def test_memory_reduction_realistic(self):
        """Test realistic memory reduction for DeepSeek."""
        # DeepSeek-V3 style: 32 heads, 128 head_dim, 512 latent
        ratio = ANELatentKVCache.compute_memory_reduction(
            num_heads=32,
            head_dim=128,
            d_latent=512,
        )
        assert ratio >= 14.0  # Should achieve 14-16x reduction


class TestANESlidingWindowCache:
    """Tests for sliding window KV cache."""

    def test_initialization(self):
        """Test sliding window cache initialization."""
        cache = ANESlidingWindowCache(
            batch_size=1,
            window_size=128,
            num_heads=8,
            head_dim=64,
            num_sink_tokens=4,
        )

        # Total size = sink + window
        assert cache.total_cache_size == 132
        assert cache.k_cache.shape == (1, 8, 132, 64)

    def test_sink_tokens_preserved(self):
        """Test that sink tokens are preserved."""
        cache = ANESlidingWindowCache(
            batch_size=1,
            window_size=32,
            num_heads=4,
            head_dim=16,
            num_sink_tokens=4,
        )

        # Add initial tokens including sink
        k = torch.randn(1, 4, 8, 16)
        v = torch.randn(1, 4, 8, 16)
        cache.update(k, v)

        # Verify sink tokens are in first positions
        assert cache.current_seq_len == 8

    def test_window_rolling(self):
        """Test window rolling behavior."""
        cache = ANESlidingWindowCache(
            batch_size=1,
            window_size=16,
            num_heads=2,
            head_dim=8,
            num_sink_tokens=4,
        )

        # Fill beyond window capacity
        for _ in range(5):
            k = torch.randn(1, 2, 8, 8)
            v = torch.randn(1, 2, 8, 8)
            cache.update(k, v)

        # Should wrap around
        assert cache.current_seq_len == 40
        # effective is sink + window
        assert cache.effective_context_length == 4 + 16

    def test_reset(self):
        """Test sliding window reset."""
        cache = ANESlidingWindowCache(
            batch_size=1,
            window_size=32,
            num_heads=4,
            head_dim=16,
        )

        k = torch.randn(1, 4, 16, 16)
        v = torch.randn(1, 4, 16, 16)
        cache.update(k, v)

        cache.reset()
        assert cache.current_seq_len == 0

    def test_effective_context_length(self):
        """Test effective context length calculation."""
        cache = ANESlidingWindowCache(
            batch_size=1,
            window_size=64,
            num_heads=4,
            head_dim=32,
            num_sink_tokens=8,
        )

        # Add 32 tokens (less than window + sink)
        k = torch.randn(1, 4, 32, 32)
        v = torch.randn(1, 4, 32, 32)
        cache.update(k, v)

        # Should report actual length since not full
        assert cache.effective_context_length == 32


class TestKVCacheConfig:
    """Tests for KV cache configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ANEKVCacheConfig()

        assert config.batch_size == 1
        assert config.max_seq_len == 8192
        assert config.num_heads == 32
        assert config.head_dim == 128
        assert config.use_fp16 is True
        assert config.alignment == 16


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
