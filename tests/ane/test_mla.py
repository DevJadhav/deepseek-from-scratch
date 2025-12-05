"""
Tests for ANE Multi-Latent Attention (MLA)

Tests for ANEMultiLatentAttention and ANEMLAConfig.
"""

import pytest
import torch

from deepseek.mlx.ane.attention.mla import ANEMLAConfig, ANEMultiLatentAttention
from deepseek.mlx.ane.attention.rope import RoPEScalingType


class TestANEMLAConfig:
    """Tests for MLA configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ANEMLAConfig()

        assert config.d_model == 4096
        assert config.num_heads == 32
        assert config.d_latent == 512
        assert config.d_rope == 64
        assert config.chunk_size == 128

    def test_d_head_computed(self):
        """Test d_head is computed from d_model and num_heads."""
        config = ANEMLAConfig(d_model=2048, num_heads=16)

        assert config.d_head == 128

    def test_d_head_explicit(self):
        """Test explicit d_head overrides computed."""
        config = ANEMLAConfig(d_model=2048, num_heads=16, d_head=64)

        assert config.d_head == 64

    def test_for_deepseek_v3_factory(self):
        """Test DeepSeek-V3 factory method."""
        config = ANEMLAConfig.for_deepseek_v3()

        assert config.d_model == 4096
        assert config.num_heads == 32
        assert config.d_latent == 512  # 4096 / 8
        assert config.rope_scaling_type == RoPEScalingType.NTK_AWARE


class TestANEMLABasic:
    """Basic tests for ANE MLA."""

    def test_initialization(self):
        """Test MLA initialization."""
        config = ANEMLAConfig(
            d_model=256,
            num_heads=4,
            d_latent=64,
            d_rope=32,
        )
        mla = ANEMultiLatentAttention(config)

        assert mla.d_model == 256
        assert mla.num_heads == 4
        assert mla.d_latent == 64

    def test_forward_shape(self):
        """Test forward pass output shape."""
        config = ANEMLAConfig(
            d_model=256,
            num_heads=4,
            d_latent=64,
            d_rope=32,
            max_seq_len=512,
        )
        mla = ANEMultiLatentAttention(config)

        x = torch.randn(2, 32, 256)  # batch, seq, d_model
        output, cached = mla(x)

        assert output.shape == x.shape
        assert cached is None  # No cache when use_cache=False

    def test_forward_with_cache(self):
        """Test forward pass with KV cache."""
        config = ANEMLAConfig(
            d_model=256,
            num_heads=4,
            d_latent=64,
            d_rope=32,
            max_seq_len=512,
        )
        mla = ANEMultiLatentAttention(config)
        mla.init_kv_cache(batch_size=2)

        x = torch.randn(2, 32, 256)
        output, cached = mla(x, use_cache=True)

        assert output.shape == x.shape
        assert cached is not None
        assert cached.shape == (2, 32, 64)  # batch, seq, d_latent


class TestANEMLACache:
    """Tests for MLA with KV cache."""

    def test_init_kv_cache(self):
        """Test KV cache initialization."""
        config = ANEMLAConfig(
            d_model=256,
            num_heads=4,
            d_latent=64,
        )
        mla = ANEMultiLatentAttention(config)

        cache = mla.init_kv_cache(batch_size=2, max_seq_len=1024)

        assert cache is not None
        assert cache.d_latent == 64
        assert cache.max_seq_len == 1024

    def test_incremental_generation(self):
        """Test incremental token generation with cache."""
        config = ANEMLAConfig(
            d_model=256,
            num_heads=4,
            d_latent=64,
            d_rope=32,
            max_seq_len=256,
        )
        mla = ANEMultiLatentAttention(config)
        mla.init_kv_cache(batch_size=1)

        # Prefill with initial context
        x1 = torch.randn(1, 16, 256)
        out1, _ = mla(x1, use_cache=True)
        assert out1.shape == (1, 16, 256)

        # Generate token by token
        x2 = torch.randn(1, 1, 256)
        out2, _ = mla(x2, use_cache=True, position_offset=16)
        assert out2.shape == (1, 1, 256)

    def test_cache_accumulates(self):
        """Test that cache accumulates tokens."""
        config = ANEMLAConfig(
            d_model=256,
            num_heads=4,
            d_latent=64,
            max_seq_len=256,
        )
        mla = ANEMultiLatentAttention(config)
        mla.init_kv_cache(batch_size=1)

        # Add first batch
        x1 = torch.randn(1, 8, 256)
        mla(x1, use_cache=True)
        assert mla.kv_cache.current_seq_len == 8

        # Add second batch
        x2 = torch.randn(1, 4, 256)
        mla(x2, use_cache=True, position_offset=8)
        assert mla.kv_cache.current_seq_len == 12


class TestANEMLAMemory:
    """Tests for MLA memory efficiency."""

    def test_memory_reduction_ratio(self):
        """Test memory reduction ratio calculation."""
        config = ANEMLAConfig(
            d_model=4096,
            num_heads=32,
            d_latent=512,
        )
        mla = ANEMultiLatentAttention(config)

        # 32 heads * 128 head_dim * 2 (K+V) = 8192
        # Latent: 512
        # Ratio: 16x
        ratio = mla.memory_reduction_ratio()
        assert ratio == 16.0

    def test_memory_reduction_realistic(self):
        """Test realistic memory reduction."""
        config = ANEMLAConfig.for_deepseek_v3()
        mla = ANEMultiLatentAttention(config)

        ratio = mla.memory_reduction_ratio()
        assert ratio >= 14.0  # Should achieve 14-16x


class TestANEMLAProjections:
    """Tests for MLA projections."""

    def test_query_projection(self):
        """Test query content and RoPE projections."""
        config = ANEMLAConfig(
            d_model=256,
            num_heads=4,
            d_latent=64,
            d_rope=32,
        )
        mla = ANEMultiLatentAttention(config)

        # Q content: d_model -> num_heads * d_head
        assert mla.q_content_proj.in_features == 256
        assert mla.q_content_proj.out_features == 4 * 64  # 256

        # Q rope: d_model -> num_heads * d_rope
        assert mla.q_rope_proj.in_features == 256
        assert mla.q_rope_proj.out_features == 4 * 32  # 128

    def test_kv_projection(self):
        """Test KV compression projections."""
        config = ANEMLAConfig(
            d_model=256,
            num_heads=4,
            d_latent=64,
            d_rope=32,
        )
        mla = ANEMultiLatentAttention(config)

        # Down projection: d_model -> d_latent
        assert mla.kv_down_proj.in_features == 256
        assert mla.kv_down_proj.out_features == 64

        # K content up: d_latent -> num_heads * d_head
        assert mla.k_content_up.in_features == 64
        assert mla.k_content_up.out_features == 4 * 64

        # K rope up: d_latent -> d_rope (shared)
        assert mla.k_rope_up.in_features == 64
        assert mla.k_rope_up.out_features == 32


class TestANEMLANumerical:
    """Tests for numerical properties."""

    def test_no_nan_output(self):
        """Test that output contains no NaN values."""
        config = ANEMLAConfig(
            d_model=256,
            num_heads=4,
            d_latent=64,
            d_rope=32,
        )
        mla = ANEMultiLatentAttention(config)

        x = torch.randn(2, 32, 256)
        output, _ = mla(x)

        assert not torch.isnan(output).any()

    def test_no_inf_output(self):
        """Test that output contains no Inf values."""
        config = ANEMLAConfig(
            d_model=256,
            num_heads=4,
            d_latent=64,
            d_rope=32,
        )
        mla = ANEMultiLatentAttention(config)

        x = torch.randn(2, 32, 256)
        output, _ = mla(x)

        assert not torch.isinf(output).any()

    def test_deterministic(self):
        """Test deterministic output in eval mode."""
        config = ANEMLAConfig(
            d_model=256,
            num_heads=4,
            d_latent=64,
        )
        mla = ANEMultiLatentAttention(config)
        mla.eval()

        x = torch.randn(1, 16, 256)

        out1, _ = mla(x)
        out2, _ = mla(x)

        assert torch.allclose(out1, out2)


class TestANEMLAFP16:
    """Tests for FP16 optimization."""

    def test_fp16_input(self):
        """Test with FP16 input."""
        config = ANEMLAConfig(
            d_model=256,
            num_heads=4,
            d_latent=64,
            use_fp16=True,
        )
        mla = ANEMultiLatentAttention(config)

        x = torch.randn(1, 16, 256, dtype=torch.float16)
        output, _ = mla(x)

        assert output.dtype == torch.float16

    def test_fp32_input_converted(self):
        """Test FP32 input is converted when use_fp16=True."""
        config = ANEMLAConfig(
            d_model=256,
            num_heads=4,
            d_latent=64,
            use_fp16=True,
        )
        mla = ANEMultiLatentAttention(config)
        mla = mla.half()  # Convert weights to FP16

        x = torch.randn(1, 16, 256, dtype=torch.float32)
        output, _ = mla(x)

        # Output should be FP32 (converted back from FP16)
        assert output.dtype == torch.float32


class TestANEMLAExtra:
    """Extra/edge case tests."""

    def test_single_token(self):
        """Test with single token input."""
        config = ANEMLAConfig(
            d_model=256,
            num_heads=4,
            d_latent=64,
        )
        mla = ANEMultiLatentAttention(config)

        x = torch.randn(1, 1, 256)
        output, _ = mla(x)

        assert output.shape == (1, 1, 256)

    def test_batch_size_one(self):
        """Test with batch size 1."""
        config = ANEMLAConfig(
            d_model=256,
            num_heads=4,
            d_latent=64,
        )
        mla = ANEMultiLatentAttention(config)

        x = torch.randn(1, 32, 256)
        output, _ = mla(x)

        assert output.shape == (1, 32, 256)

    def test_extra_repr(self):
        """Test extra_repr for debugging."""
        config = ANEMLAConfig(
            d_model=4096,
            num_heads=32,
            d_latent=512,
        )
        mla = ANEMultiLatentAttention(config)

        repr_str = mla.extra_repr()
        assert "d_model=4096" in repr_str
        assert "num_heads=32" in repr_str
        assert "d_latent=512" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
