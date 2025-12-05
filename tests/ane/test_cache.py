"""
Tests for Phase 5: KV Cache Optimization

This module tests the unified memory KV cache, KV split quantization,
and latent cache implementations for ANE optimization.
"""

import platform
import sys
from pathlib import Path

import pytest
import torch

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Skip all tests if not on Apple Silicon
IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"

from deepseek.mlx.ane.KVcache import (
    ComputeUnit,
    KVQuantType,
    KVSplitCache,
    KVSplitConfig,
    KVSplitQuantizer,
    LatentCacheConfig,
    LatentCacheStats,
    LatentKVCacheUnified,
    LatentQuantType,
    UnifiedMemoryConfig,
    UnifiedMemoryKVCache,
    UnifiedMemoryStats,
    check_unified_memory_available,
    dequantize_keys_int8,
    dequantize_values_int4,
    quantize_keys_int8,
    quantize_values_int4,
    zero_copy_transfer,
)


class TestUnifiedMemoryKVCache:
    """Tests for UnifiedMemoryKVCache."""

    @pytest.fixture
    def config(self):
        return UnifiedMemoryConfig(
            batch_size=2,
            max_seq_len=128,
            num_heads=4,
            head_dim=32,
            num_layers=2,
            use_fp16=True,
        )

    @pytest.fixture
    def cache(self, config):
        return UnifiedMemoryKVCache(config)

    def test_init(self, cache, config):
        """Test cache initialization."""
        assert cache.batch_size == config.batch_size
        assert cache.max_seq_len == config.max_seq_len
        assert cache.num_heads == config.num_heads
        assert cache.head_dim == config.head_dim
        assert cache.num_layers == config.num_layers
        assert len(cache.k_caches) == config.num_layers
        assert len(cache.v_caches) == config.num_layers
        assert cache.current_seq_len == 0

    def test_update_sequential(self, cache, config):
        """Test sequential cache update."""
        batch_size = 2
        seq_len = 16
        k = torch.randn(batch_size, config.num_heads, seq_len, config.head_dim)
        v = torch.randn(batch_size, config.num_heads, seq_len, config.head_dim)

        k_cached, v_cached = cache.update(layer_idx=0, k=k, v=v)

        assert k_cached.shape == (batch_size, config.num_heads, seq_len, config.head_dim)
        assert v_cached.shape == (batch_size, config.num_heads, seq_len, config.head_dim)
        assert cache.current_seq_len == seq_len

    def test_update_multi_layer(self, config):
        """Test updating multiple layers."""
        # Create a fresh cache for this test
        cache = UnifiedMemoryKVCache(config)
        batch_size = 2
        seq_len = 16
        k = torch.randn(batch_size, config.num_heads, seq_len, config.head_dim)
        v = torch.randn(batch_size, config.num_heads, seq_len, config.head_dim)

        # Update all layers with same data (simulating forward pass)
        for layer_idx in range(config.num_layers):
            k_cached, v_cached = cache.update(layer_idx=layer_idx, k=k, v=v)
            # Each layer should have same seq_len
            assert k_cached.shape == (batch_size, config.num_heads, seq_len, config.head_dim)

    def test_update_incremental(self, cache, config):
        """Test incremental cache updates."""
        batch_size = 2
        seq_len = 8

        # First update
        k1 = torch.randn(batch_size, config.num_heads, seq_len, config.head_dim)
        v1 = torch.randn(batch_size, config.num_heads, seq_len, config.head_dim)
        cache.update(layer_idx=0, k=k1, v=v1)
        assert cache.current_seq_len == seq_len

        # Second update (incremental)
        k2 = torch.randn(batch_size, config.num_heads, 4, config.head_dim)
        v2 = torch.randn(batch_size, config.num_heads, 4, config.head_dim)
        k_cached, v_cached = cache.update(layer_idx=0, k=k2, v=v2)

        assert k_cached.shape == (batch_size, config.num_heads, seq_len + 4, config.head_dim)
        assert cache.current_seq_len == seq_len + 4

    def test_get_cached_kv(self, cache, config):
        """Test retrieving cached KV."""
        batch_size = 2
        seq_len = 16
        k = torch.randn(batch_size, config.num_heads, seq_len, config.head_dim)
        v = torch.randn(batch_size, config.num_heads, seq_len, config.head_dim)

        cache.update(layer_idx=0, k=k, v=v)
        k_cached, v_cached = cache.get_cached_kv(layer_idx=0)

        assert k_cached.shape == (batch_size, config.num_heads, seq_len, config.head_dim)
        assert v_cached.shape == (batch_size, config.num_heads, seq_len, config.head_dim)

    def test_reset(self, cache, config):
        """Test cache reset."""
        batch_size = 2
        seq_len = 16
        k = torch.randn(batch_size, config.num_heads, seq_len, config.head_dim)
        v = torch.randn(batch_size, config.num_heads, seq_len, config.head_dim)

        cache.update(layer_idx=0, k=k, v=v)
        assert cache.current_seq_len == seq_len

        cache.reset()
        assert cache.current_seq_len == 0

    def test_get_stats(self, cache, config):
        """Test memory statistics."""
        stats = cache.get_stats()
        assert isinstance(stats, UnifiedMemoryStats)
        assert stats.total_bytes > 0
        assert stats.num_layers == config.num_layers
        assert stats.max_seq_len == config.max_seq_len

    def test_memory_usage_bytes(self, cache, config):
        """Test memory usage calculation."""
        memory = cache.memory_usage_bytes()
        expected = (
            config.batch_size * config.num_heads * config.max_seq_len * config.head_dim *
            2 * 2 * config.num_layers  # K and V, FP16 (2 bytes)
        )
        assert memory == expected

    def test_exceeds_max_seq_len(self, cache, config):
        """Test error when exceeding max sequence length."""
        k = torch.randn(2, config.num_heads, config.max_seq_len + 10, config.head_dim)
        v = torch.randn(2, config.num_heads, config.max_seq_len + 10, config.head_dim)

        with pytest.raises(ValueError, match="exceeds max_seq_len"):
            cache.update(layer_idx=0, k=k, v=v)

    def test_invalid_layer_idx(self, cache, config):
        """Test error for invalid layer index."""
        k = torch.randn(2, config.num_heads, 16, config.head_dim)
        v = torch.randn(2, config.num_heads, 16, config.head_dim)

        with pytest.raises(ValueError, match="Layer index"):
            cache.update(layer_idx=config.num_layers + 1, k=k, v=v)


class TestKVSplitQuantizer:
    """Tests for KV Split Quantization."""

    @pytest.fixture
    def quantizer(self):
        config = KVSplitConfig(quant_type=KVQuantType.K8V4)
        return KVSplitQuantizer(config)

    def test_quantize_keys_int8(self):
        """Test INT8 key quantization."""
        k = torch.randn(2, 4, 16, 32)  # batch, heads, seq, head_dim
        k_quant, scale = quantize_keys_int8(k, per_head=True)

        assert k_quant.dtype == torch.int8
        assert k_quant.shape == k.shape
        assert scale.shape[0] == 2  # batch
        assert scale.shape[1] == 4  # heads

    def test_dequantize_keys_int8(self):
        """Test INT8 key dequantization."""
        k = torch.randn(2, 4, 16, 32)
        k_quant, scale = quantize_keys_int8(k, per_head=True)
        k_dequant = dequantize_keys_int8(k_quant, scale)

        assert k_dequant.shape == k.shape
        assert k_dequant.dtype == torch.float16

        # Check reconstruction error is reasonable
        error = (k.half() - k_dequant).abs().mean()
        assert error < 0.1  # Should be small for INT8

    def test_quantize_values_int4(self):
        """Test INT4 value quantization."""
        v = torch.randn(2, 4, 16, 32)  # batch, heads, seq, head_dim
        v_quant, scale = quantize_values_int4(v, per_head=True)

        assert v_quant.dtype == torch.int8  # Packed INT4
        # Packed dimension should be half (or half + 1 for odd)
        expected_packed_dim = 32 // 2
        assert v_quant.shape[-1] == expected_packed_dim

    def test_dequantize_values_int4(self):
        """Test INT4 value dequantization."""
        v = torch.randn(2, 4, 16, 32)
        v_quant, scale = quantize_values_int4(v, per_head=True)
        v_dequant = dequantize_values_int4(v_quant, scale, v.shape)

        assert v_dequant.shape == v.shape
        assert v_dequant.dtype == torch.float16

        # Check reconstruction error (INT4 has higher error)
        error = (v.half() - v_dequant).abs().mean()
        assert error < 0.5  # Higher tolerance for INT4

    def test_quantizer_k8v4(self, quantizer):
        """Test K8V4 quantization through quantizer."""
        k = torch.randn(2, 4, 16, 32)
        v = torch.randn(2, 4, 16, 32)

        quantized = quantizer.quantize(k, v)

        assert quantized.k_data.dtype == torch.int8
        assert quantized.v_data.dtype == torch.int8  # Packed INT4
        assert quantized.quant_type == KVQuantType.K8V4

    def test_quantizer_roundtrip(self, quantizer):
        """Test quantize-dequantize roundtrip."""
        k = torch.randn(2, 4, 16, 32)
        v = torch.randn(2, 4, 16, 32)

        quantized = quantizer.quantize(k, v)
        k_dequant, v_dequant = quantizer.dequantize(quantized)

        assert k_dequant.shape == k.shape
        assert v_dequant.shape == v.shape

        # Check reconstruction errors
        k_error = (k.half() - k_dequant).abs().mean()
        v_error = (v.half() - v_dequant).abs().mean()
        assert k_error < 0.1  # INT8 should be accurate
        assert v_error < 0.5  # INT4 has more error

    def test_compression_ratio(self, quantizer):
        """Test compression ratio calculation."""
        ratio = quantizer.compute_compression_ratio()
        # K8V4: 4 bytes (FP16 K+V) / 1.5 bytes (INT8 K + INT4 V) ≈ 2.67
        assert 2.5 < ratio < 2.8


class TestKVSplitCache:
    """Tests for KVSplitCache."""

    @pytest.fixture
    def cache(self):
        config = KVSplitConfig(quant_type=KVQuantType.K8V4)
        return KVSplitCache(
            batch_size=2,
            max_seq_len=128,
            num_heads=4,
            head_dim=32,
            config=config,
        )

    def test_init(self, cache):
        """Test cache initialization."""
        assert cache.batch_size == 2
        assert cache.max_seq_len == 128
        assert cache.num_heads == 4
        assert cache.head_dim == 32
        assert cache.current_seq_len == 0

    def test_update(self, cache):
        """Test cache update with automatic quantization."""
        k = torch.randn(2, 4, 16, 32)
        v = torch.randn(2, 4, 16, 32)

        k_cached, v_cached = cache.update(k, v)

        assert k_cached.shape == (2, 4, 16, 32)
        assert v_cached.shape == (2, 4, 16, 32)
        assert cache.current_seq_len == 16

    def test_update_incremental(self, cache):
        """Test incremental updates."""
        k1 = torch.randn(2, 4, 8, 32)
        v1 = torch.randn(2, 4, 8, 32)
        cache.update(k1, v1)

        k2 = torch.randn(2, 4, 4, 32)
        v2 = torch.randn(2, 4, 4, 32)
        k_cached, v_cached = cache.update(k2, v2)

        assert k_cached.shape == (2, 4, 12, 32)
        assert cache.current_seq_len == 12

    def test_get_cached_kv(self, cache):
        """Test retrieving dequantized cached KV."""
        k = torch.randn(2, 4, 16, 32)
        v = torch.randn(2, 4, 16, 32)
        cache.update(k, v)

        k_cached, v_cached = cache.get_cached_kv()

        assert k_cached.shape == (2, 4, 16, 32)
        assert v_cached.shape == (2, 4, 16, 32)
        assert k_cached.dtype == torch.float16
        assert v_cached.dtype == torch.float16

    def test_reset(self, cache):
        """Test cache reset."""
        k = torch.randn(2, 4, 16, 32)
        v = torch.randn(2, 4, 16, 32)
        cache.update(k, v)

        cache.reset()
        assert cache.current_seq_len == 0

    def test_memory_usage(self, cache):
        """Test memory usage calculation."""
        memory = cache.memory_usage_bytes()
        assert memory > 0

    def test_compression_ratio(self, cache):
        """Test compression ratio."""
        ratio = cache.compression_ratio()
        assert ratio > 2.0  # K8V4 should provide good compression


class TestLatentKVCacheUnified:
    """Tests for LatentKVCacheUnified."""

    @pytest.fixture
    def config(self):
        return LatentCacheConfig(
            batch_size=2,
            max_seq_len=128,
            d_latent=64,
            num_layers=2,
            use_quantization=True,
            quant_type=LatentQuantType.INT8,
        )

    @pytest.fixture
    def cache(self, config):
        return LatentKVCacheUnified(config)

    def test_init(self, cache, config):
        """Test cache initialization."""
        assert cache.batch_size == config.batch_size
        assert cache.max_seq_len == config.max_seq_len
        assert cache.d_latent == config.d_latent
        assert cache.num_layers == config.num_layers
        assert len(cache.latent_caches) == config.num_layers

    def test_update(self, cache, config):
        """Test latent cache update."""
        c_kv = torch.randn(2, 16, config.d_latent)
        latent_cached = cache.update(layer_idx=0, c_kv=c_kv)

        assert latent_cached.shape == (2, 16, config.d_latent)
        assert cache.current_seq_len == 16

    def test_update_multi_layer(self, cache, config):
        """Test updating multiple layers."""
        c_kv = torch.randn(2, 16, config.d_latent)

        for layer_idx in range(config.num_layers):
            latent_cached = cache.update(layer_idx=layer_idx, c_kv=c_kv)
            assert latent_cached.shape == (2, 16, config.d_latent)

    def test_update_incremental(self, cache, config):
        """Test incremental updates."""
        c_kv1 = torch.randn(2, 8, config.d_latent)
        cache.update(layer_idx=0, c_kv=c_kv1)

        c_kv2 = torch.randn(2, 4, config.d_latent)
        latent_cached = cache.update(layer_idx=0, c_kv=c_kv2)

        assert latent_cached.shape == (2, 12, config.d_latent)
        assert cache.current_seq_len == 12

    def test_get_cached_latent(self, cache, config):
        """Test retrieving cached latent."""
        c_kv = torch.randn(2, 16, config.d_latent)
        cache.update(layer_idx=0, c_kv=c_kv)

        latent_cached = cache.get_cached_latent(layer_idx=0)
        assert latent_cached.shape == (2, 16, config.d_latent)

    def test_reset(self, cache, config):
        """Test cache reset."""
        c_kv = torch.randn(2, 16, config.d_latent)
        cache.update(layer_idx=0, c_kv=c_kv)

        cache.reset()
        assert cache.current_seq_len == 0

    def test_get_stats(self, cache, config):
        """Test memory statistics."""
        stats = cache.get_stats()
        assert isinstance(stats, LatentCacheStats)
        assert stats.total_bytes > 0
        assert stats.num_layers == config.num_layers
        assert stats.compression_ratio > 1.0

    def test_quantization_roundtrip(self, cache, config):
        """Test quantization maintains reasonable accuracy."""
        c_kv = torch.randn(2, 16, config.d_latent)
        latent_cached = cache.update(layer_idx=0, c_kv=c_kv)

        # Move to same device for comparison
        c_kv_compare = c_kv.half().to(latent_cached.device)

        # Check reconstruction error
        error = (c_kv_compare - latent_cached).abs().mean()
        assert error < 0.1  # Should be small for INT8

    def test_no_quantization(self):
        """Test cache without quantization."""
        config = LatentCacheConfig(
            batch_size=2,
            max_seq_len=128,
            d_latent=64,
            num_layers=2,
            use_quantization=False,
            quant_type=LatentQuantType.NONE,
        )
        cache = LatentKVCacheUnified(config)

        c_kv = torch.randn(2, 16, 64)
        latent_cached = cache.update(layer_idx=0, c_kv=c_kv)

        # Move to same device for comparison
        c_kv_compare = c_kv.half().to(latent_cached.device)

        # Without quantization, should be exact (up to FP16 precision)
        error = (c_kv_compare - latent_cached).abs().mean()
        assert error < 1e-3

    def test_int4_quantization(self):
        """Test INT4 quantization."""
        config = LatentCacheConfig(
            batch_size=2,
            max_seq_len=128,
            d_latent=64,
            num_layers=2,
            use_quantization=True,
            quant_type=LatentQuantType.INT4,
        )
        cache = LatentKVCacheUnified(config)

        c_kv = torch.randn(2, 16, 64)
        latent_cached = cache.update(layer_idx=0, c_kv=c_kv)

        # Move to same device for comparison
        c_kv_compare = c_kv.half().to(latent_cached.device)

        # INT4 has more error but should still be usable
        error = (c_kv_compare - latent_cached).abs().mean()
        assert error < 0.5


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_check_unified_memory_available(self):
        """Test unified memory detection."""
        result = check_unified_memory_available()
        if IS_APPLE_SILICON:
            assert result is True
        else:
            assert result is False

    def test_zero_copy_transfer_cpu(self):
        """Test zero-copy transfer to CPU."""
        tensor = torch.randn(2, 4, 16, 32)
        result = zero_copy_transfer(tensor, ComputeUnit.CPU)
        assert result.device.type == "cpu"

    @pytest.mark.skipif(not IS_APPLE_SILICON, reason="Requires Apple Silicon")
    def test_zero_copy_transfer_mps(self):
        """Test zero-copy transfer to MPS (GPU)."""
        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available")

        tensor = torch.randn(2, 4, 16, 32)
        result = zero_copy_transfer(tensor, ComputeUnit.GPU)
        assert result.device.type == "mps"


class TestCompressionRatios:
    """Tests for compression ratio calculations."""

    def test_latent_memory_reduction(self):
        """Test latent memory reduction calculation."""
        # Standard: 32 heads × 128 dim × 2 (K+V) × 2 bytes = 16KB per token
        # Latent FP16: 512 × 2 bytes = 1KB per token → 16x
        reduction = LatentKVCacheUnified.compute_latent_memory_reduction(
            num_heads=32,
            head_dim=128,
            d_latent=512,
            quant_type=LatentQuantType.NONE,
        )
        assert 14 < reduction < 18

        # Latent INT8: 512 × 1 byte + scale = ~516B per token → ~32x
        reduction_int8 = LatentKVCacheUnified.compute_latent_memory_reduction(
            num_heads=32,
            head_dim=128,
            d_latent=512,
            quant_type=LatentQuantType.INT8,
        )
        assert reduction_int8 > reduction

        # Latent INT4: 512 × 0.5 byte + scale = ~260B per token → ~64x
        reduction_int4 = LatentKVCacheUnified.compute_latent_memory_reduction(
            num_heads=32,
            head_dim=128,
            d_latent=512,
            quant_type=LatentQuantType.INT4,
        )
        assert reduction_int4 > reduction_int8

    def test_kvsplit_compression_ratios(self):
        """Test KV split compression ratios."""
        # K8V4: ~2.67x compression
        config_k8v4 = KVSplitConfig(quant_type=KVQuantType.K8V4)
        quantizer = KVSplitQuantizer(config_k8v4)
        ratio = quantizer.compute_compression_ratio()
        assert 2.5 < ratio < 2.8

        # K8V8: 2x compression
        config_k8v8 = KVSplitConfig(quant_type=KVQuantType.K8V8)
        quantizer_k8v8 = KVSplitQuantizer(config_k8v8)
        ratio_k8v8 = quantizer_k8v8.compute_compression_ratio()
        assert 1.9 < ratio_k8v8 < 2.1

        # K4V4: 4x compression
        config_k4v4 = KVSplitConfig(quant_type=KVQuantType.K4V4)
        quantizer_k4v4 = KVSplitQuantizer(config_k4v4)
        ratio_k4v4 = quantizer_k4v4.compute_compression_ratio()
        assert 3.8 < ratio_k4v4 < 4.2


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_cache_access(self):
        """Test accessing empty cache."""
        config = UnifiedMemoryConfig(
            batch_size=1,
            max_seq_len=128,
            num_heads=4,
            head_dim=32,
            num_layers=1,
        )
        cache = UnifiedMemoryKVCache(config)

        k_cached, v_cached = cache.get_cached_kv(layer_idx=0)
        assert k_cached.shape == (1, 4, 0, 32)
        assert v_cached.shape == (1, 4, 0, 32)

    def test_kvsplit_empty_cache(self):
        """Test KVSplitCache with empty access."""
        cache = KVSplitCache(
            batch_size=1,
            max_seq_len=128,
            num_heads=4,
            head_dim=32,
        )

        k_cached, v_cached = cache.get_cached_kv()
        assert k_cached.shape == (1, 4, 0, 32)
        assert v_cached.shape == (1, 4, 0, 32)

    def test_single_token_update(self):
        """Test updating with single token."""
        config = UnifiedMemoryConfig(
            batch_size=1,
            max_seq_len=128,
            num_heads=4,
            head_dim=32,
            num_layers=1,
        )
        cache = UnifiedMemoryKVCache(config)

        k = torch.randn(1, 4, 1, 32)
        v = torch.randn(1, 4, 1, 32)

        k_cached, v_cached = cache.update(layer_idx=0, k=k, v=v)
        assert k_cached.shape == (1, 4, 1, 32)
        assert cache.current_seq_len == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
