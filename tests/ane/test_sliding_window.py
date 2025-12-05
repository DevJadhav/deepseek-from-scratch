"""
Tests for Sliding Window KV Cache

This module tests the sliding window cache implementations for
extreme context length handling on Apple Neural Engine.
"""

import pytest
import torch

from deepseek.mlx.ane.KVcache.sliding_window import (
    SlidingWindowConfig,
    SlidingWindowKVCache,
    SlidingWindowStats,
    StreamingSlidingWindowCache,
    WindowEvictionPolicy,
    create_sliding_window_cache,
    estimate_memory_savings,
)


class TestSlidingWindowConfig:
    """Tests for SlidingWindowConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SlidingWindowConfig()

        assert config.window_size == 4096
        assert config.sink_tokens == 4
        assert config.batch_size == 1
        assert config.num_heads == 32
        assert config.head_dim == 128
        assert config.num_layers == 32
        assert config.dtype == torch.float16
        assert config.eviction_policy == WindowEvictionPolicy.FIFO

    def test_custom_config(self):
        """Test custom configuration."""
        config = SlidingWindowConfig(
            window_size=2048,
            sink_tokens=8,
            num_heads=16,
            head_dim=64,
            num_layers=12,
        )

        assert config.window_size == 2048
        assert config.sink_tokens == 8
        assert config.total_window_size == 2056

    def test_total_window_size(self):
        """Test total window size calculation."""
        config = SlidingWindowConfig(window_size=1024, sink_tokens=4)
        assert config.total_window_size == 1028

    def test_memory_bytes_per_layer(self):
        """Test memory calculation per layer."""
        config = SlidingWindowConfig(
            window_size=256,
            sink_tokens=4,
            batch_size=1,
            num_heads=8,
            head_dim=64,
            dtype=torch.float16,
        )

        # Total size = 256 + 4 = 260
        # K bytes = 1 * 8 * 260 * 64 * 2 = 266,240
        # V bytes = same
        # Total per layer = 532,480
        expected = 2 * 1 * 8 * 260 * 64 * 2
        assert config.memory_bytes_per_layer == expected


class TestSlidingWindowKVCache:
    """Tests for SlidingWindowKVCache."""

    @pytest.fixture
    def small_config(self):
        """Create a small config for testing."""
        return SlidingWindowConfig(
            window_size=16,
            sink_tokens=4,
            batch_size=1,
            num_heads=4,
            head_dim=32,
            num_layers=2,
            dtype=torch.float32,  # Use float32 for easier debugging
        )

    def test_init(self, small_config):
        """Test cache initialization."""
        cache = SlidingWindowKVCache(small_config)

        assert len(cache.k_caches) == 2
        assert len(cache.v_caches) == 2
        assert cache.k_caches[0].shape == (1, 4, 20, 32)  # 16 + 4 = 20
        assert cache.total_tokens_seen == 0
        assert cache.tokens_evicted == 0

    def test_update_single_token(self, small_config):
        """Test updating with single token."""
        cache = SlidingWindowKVCache(small_config)

        k = torch.randn(1, 4, 1, 32)
        v = torch.randn(1, 4, 1, 32)

        cached_k, cached_v = cache.update(0, k, v)

        assert cached_k.shape == (1, 4, 1, 32)
        assert cached_v.shape == (1, 4, 1, 32)
        assert cache.total_tokens_seen == 1

    def test_update_sequential(self, small_config):
        """Test sequential token updates."""
        cache = SlidingWindowKVCache(small_config)

        # Add tokens one by one
        for i in range(10):
            k = torch.full((1, 4, 1, 32), float(i))
            v = torch.full((1, 4, 1, 32), float(i))
            cache.update(0, k, v)

        assert cache.total_tokens_seen == 10
        assert cache.tokens_evicted == 0  # Window not full yet

        cached_k, cached_v = cache.get_cached_kv(0)
        assert cached_k.shape == (1, 4, 10, 32)

    def test_update_batch(self, small_config):
        """Test batch update (multiple tokens at once)."""
        cache = SlidingWindowKVCache(small_config)

        k = torch.randn(1, 4, 8, 32)
        v = torch.randn(1, 4, 8, 32)

        cached_k, cached_v = cache.update(0, k, v)

        assert cache.total_tokens_seen == 8
        assert cached_k.shape == (1, 4, 8, 32)

    def test_sliding_window_eviction(self, small_config):
        """Test that tokens are evicted when window is full."""
        cache = SlidingWindowKVCache(small_config)

        # Fill the cache completely (20 tokens = 4 sink + 16 window)
        k = torch.randn(1, 4, 20, 32)
        v = torch.randn(1, 4, 20, 32)
        cache.update(0, k, v)

        assert cache.total_tokens_seen == 20
        assert cache.tokens_evicted == 0

        # Add one more token - should trigger eviction
        k_new = torch.randn(1, 4, 1, 32)
        v_new = torch.randn(1, 4, 1, 32)
        cache.update(0, k_new, v_new)

        assert cache.total_tokens_seen == 21
        assert cache.tokens_evicted == 1

        # Cache should still be bounded
        cached_k, cached_v = cache.get_cached_kv(0)
        assert cached_k.shape[2] == 20

    def test_sink_tokens_preserved(self, small_config):
        """Test that sink tokens are never evicted."""
        cache = SlidingWindowKVCache(small_config)

        # Fill cache with identifiable tokens
        # Sink tokens (first 4) have value 100 + i
        # Window tokens have value i
        for i in range(4):
            k = torch.full((1, 4, 1, 32), 100.0 + i)
            v = torch.full((1, 4, 1, 32), 100.0 + i)
            cache.update(0, k, v)

        # Fill window
        for i in range(16):
            k = torch.full((1, 4, 1, 32), float(i))
            v = torch.full((1, 4, 1, 32), float(i))
            cache.update(0, k, v)

        # Add more tokens to trigger sliding
        for i in range(10):
            k = torch.full((1, 4, 1, 32), 1000.0 + i)
            v = torch.full((1, 4, 1, 32), 1000.0 + i)
            cache.update(0, k, v)

        # Check sink tokens are preserved
        cached_k, _ = cache.get_cached_kv(0)

        # First 4 positions should still have sink token values
        for i in range(4):
            assert torch.allclose(
                cached_k[0, 0, i, 0],
                torch.tensor(100.0 + i),
            ), f"Sink token {i} was modified"

    def test_multi_layer_update(self, small_config):
        """Test that all layers can be updated correctly."""
        cache = SlidingWindowKVCache(small_config)

        k = torch.randn(1, 4, 8, 32)
        v = torch.randn(1, 4, 8, 32)

        # Update both layers with same KV
        for layer_idx in range(2):
            cache.update(layer_idx, k, v)

        # Check both layers have same content
        k0, v0 = cache.get_cached_kv(0)
        k1, v1 = cache.get_cached_kv(1)

        assert k0.shape == k1.shape == (1, 4, 8, 32)
        assert torch.allclose(k0, k1)
        assert torch.allclose(v0, v1)

    def test_reset(self, small_config):
        """Test cache reset."""
        cache = SlidingWindowKVCache(small_config)

        # Add some tokens
        k = torch.randn(1, 4, 10, 32)
        v = torch.randn(1, 4, 10, 32)
        cache.update(0, k, v)

        assert cache.total_tokens_seen == 10

        # Reset
        cache.reset()

        assert cache.total_tokens_seen == 0
        assert cache.tokens_evicted == 0
        assert cache.current_pos == 0

        # Cache should be empty
        cached_k, cached_v = cache.get_cached_kv(0)
        assert cached_k.shape[2] == 0

    def test_get_stats(self, small_config):
        """Test statistics reporting."""
        cache = SlidingWindowKVCache(small_config)

        # Add some tokens
        k = torch.randn(1, 4, 15, 32)
        v = torch.randn(1, 4, 15, 32)
        cache.update(0, k, v)

        stats = cache.get_stats()

        assert isinstance(stats, SlidingWindowStats)
        assert stats.total_tokens_seen == 15
        assert stats.tokens_in_window == 15
        assert stats.tokens_evicted == 0
        assert stats.window_size == 16
        assert stats.sink_tokens == 4
        assert 0 < stats.utilization < 1

    def test_get_attention_mask(self, small_config):
        """Test attention mask generation."""
        cache = SlidingWindowKVCache(small_config)

        # Add tokens
        k = torch.randn(1, 4, 10, 32)
        v = torch.randn(1, 4, 10, 32)
        cache.update(0, k, v)

        mask = cache.get_attention_mask(query_len=1)

        assert mask.shape == (1, 1, 1, 10)
        assert mask.dtype == torch.bool

    def test_invalid_layer_idx(self, small_config):
        """Test error handling for invalid layer index."""
        cache = SlidingWindowKVCache(small_config)

        k = torch.randn(1, 4, 1, 32)
        v = torch.randn(1, 4, 1, 32)

        with pytest.raises(ValueError):
            cache.update(-1, k, v)

        with pytest.raises(ValueError):
            cache.update(10, k, v)

    def test_device_transfer(self, small_config):
        """Test moving cache to different device."""
        cache = SlidingWindowKVCache(small_config)

        k = torch.randn(1, 4, 5, 32)
        v = torch.randn(1, 4, 5, 32)
        cache.update(0, k, v)

        # Transfer to CPU (already on CPU, but tests the method)
        cache = cache.to(torch.device("cpu"))

        cached_k, cached_v = cache.get_cached_kv(0)
        assert cached_k.device == torch.device("cpu")


class TestStreamingSlidingWindowCache:
    """Tests for StreamingSlidingWindowCache."""

    @pytest.fixture
    def small_config(self):
        """Create a small config for testing."""
        return SlidingWindowConfig(
            window_size=16,
            sink_tokens=4,
            batch_size=1,
            num_heads=4,
            head_dim=32,
            num_layers=2,
            dtype=torch.float32,
        )

    def test_checkpoint_restore(self, small_config):
        """Test checkpoint and restore functionality."""
        cache = StreamingSlidingWindowCache(small_config)

        # Add tokens
        k1 = torch.randn(1, 4, 10, 32)
        v1 = torch.randn(1, 4, 10, 32)
        cache.update(0, k1, v1)

        # Checkpoint
        state = cache.checkpoint()

        # Add more tokens
        k2 = torch.randn(1, 4, 5, 32)
        v2 = torch.randn(1, 4, 5, 32)
        cache.update(0, k2, v2)

        assert cache.total_tokens_seen == 15

        # Restore
        cache.restore(state)

        assert cache.total_tokens_seen == 10

    def test_speculative_update(self, small_config):
        """Test speculative update with partial acceptance."""
        cache = StreamingSlidingWindowCache(small_config)

        # Add initial tokens
        k_init = torch.randn(1, 4, 5, 32)
        v_init = torch.randn(1, 4, 5, 32)
        cache.update(0, k_init, v_init)

        # Speculative tokens (only 2 accepted)
        k_spec = torch.randn(1, 4, 4, 32)
        v_spec = torch.randn(1, 4, 4, 32)
        cache.speculative_update(0, k_spec, v_spec, num_accepted=2)

        assert cache.total_tokens_seen == 7  # 5 + 2 accepted

    def test_speculative_update_none_accepted(self, small_config):
        """Test speculative update with no accepted tokens."""
        cache = StreamingSlidingWindowCache(small_config)

        k_init = torch.randn(1, 4, 5, 32)
        v_init = torch.randn(1, 4, 5, 32)
        cache.update(0, k_init, v_init)

        k_spec = torch.randn(1, 4, 4, 32)
        v_spec = torch.randn(1, 4, 4, 32)
        cache.speculative_update(0, k_spec, v_spec, num_accepted=0)

        # Should not change
        assert cache.total_tokens_seen == 5

    def test_rollback(self, small_config):
        """Test rollback functionality."""
        cache = StreamingSlidingWindowCache(small_config)

        # Add tokens
        k = torch.randn(1, 4, 10, 32)
        v = torch.randn(1, 4, 10, 32)
        cache.update(0, k, v)

        assert cache.total_tokens_seen == 10

        # Rollback 3 tokens
        cache.rollback(3)

        assert cache.total_tokens_seen == 7


class TestFactoryFunction:
    """Tests for create_sliding_window_cache factory function."""

    def test_create_basic_cache(self):
        """Test creating basic sliding window cache."""
        cache = create_sliding_window_cache(
            window_size=512,
            sink_tokens=4,
            num_heads=8,
            head_dim=64,
            num_layers=6,
        )

        assert isinstance(cache, SlidingWindowKVCache)
        assert cache.window_size == 512
        assert cache.sink_tokens == 4

    def test_create_streaming_cache(self):
        """Test creating streaming cache."""
        cache = create_sliding_window_cache(
            window_size=512,
            sink_tokens=4,
            streaming=True,
        )

        assert isinstance(cache, StreamingSlidingWindowCache)


class TestMemorySavingsEstimate:
    """Tests for memory savings estimation."""

    def test_memory_savings_calculation(self):
        """Test memory savings calculation."""
        result = estimate_memory_savings(
            max_seq_len=8192,
            window_size=2048,
            sink_tokens=4,
            num_heads=32,
            head_dim=128,
            num_layers=32,
            dtype=torch.float16,
        )

        assert "full_cache_mb" in result
        assert "sliding_cache_mb" in result
        assert "savings_mb" in result
        assert "savings_ratio" in result

        # Savings should be significant
        assert result["savings_ratio"] > 2.0

    def test_memory_savings_extreme_context(self):
        """Test memory savings for extreme context (128K)."""
        result = estimate_memory_savings(
            max_seq_len=131072,  # 128K
            window_size=4096,
            sink_tokens=4,
            num_heads=32,
            head_dim=128,
            num_layers=32,
        )

        # Should achieve ~32x memory reduction
        assert result["savings_ratio"] > 20.0


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_cache(self):
        """Test operations on empty cache."""
        config = SlidingWindowConfig(
            window_size=16,
            sink_tokens=4,
            num_heads=4,
            head_dim=32,
            num_layers=2,
        )
        cache = SlidingWindowKVCache(config)

        # Get from empty cache
        k, v = cache.get_cached_kv(0)
        assert k.shape[2] == 0
        assert v.shape[2] == 0

    def test_single_token_window(self):
        """Test with minimum window size."""
        config = SlidingWindowConfig(
            window_size=1,
            sink_tokens=1,
            num_heads=4,
            head_dim=32,
            num_layers=2,
        )
        cache = SlidingWindowKVCache(config)

        # Add multiple tokens
        for i in range(5):
            k = torch.full((1, 4, 1, 32), float(i))
            v = torch.full((1, 4, 1, 32), float(i))
            cache.update(0, k, v)

        # Should only have 2 tokens (1 sink + 1 window)
        cached_k, _ = cache.get_cached_kv(0)
        assert cached_k.shape[2] == 2

    def test_zero_sink_tokens(self):
        """Test with no sink tokens."""
        config = SlidingWindowConfig(
            window_size=8,
            sink_tokens=0,
            num_heads=4,
            head_dim=32,
            num_layers=2,
        )
        cache = SlidingWindowKVCache(config)

        # Add tokens
        for i in range(20):
            k = torch.full((1, 4, 1, 32), float(i))
            v = torch.full((1, 4, 1, 32), float(i))
            cache.update(0, k, v)

        # Should only have window_size tokens
        cached_k, _ = cache.get_cached_kv(0)
        assert cached_k.shape[2] == 8

    def test_large_batch_update(self):
        """Test updating with batch larger than window."""
        config = SlidingWindowConfig(
            window_size=16,
            sink_tokens=4,
            num_heads=4,
            head_dim=32,
            num_layers=2,
        )
        cache = SlidingWindowKVCache(config)

        # Add more tokens than total window size
        k = torch.randn(1, 4, 30, 32)
        v = torch.randn(1, 4, 30, 32)
        cache.update(0, k, v)

        # Should be bounded by total window size
        cached_k, _ = cache.get_cached_kv(0)
        assert cached_k.shape[2] == 20  # 16 + 4
