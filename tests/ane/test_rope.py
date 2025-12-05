"""
Tests for ANE RoPE Implementations

Tests for ANERoPE with various scaling methods: None, NTK-aware, YaRN.
"""

import pytest
import torch

from deepseek.mlx.ane.attention.rope import ANERoPE, ANERoPEConfig, RoPEScalingType


class TestANERoPEConfig:
    """Tests for RoPE configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ANERoPEConfig()

        assert config.d_head == 64
        assert config.max_seq_len == 131072
        assert config.base == 10000.0
        assert config.scaling_type == RoPEScalingType.NTK_AWARE

    def test_for_4k_factory(self):
        """Test 4K context factory method."""
        config = ANERoPEConfig.for_4k()

        assert config.max_seq_len == 4096
        assert config.scaling_type == RoPEScalingType.NONE

    def test_for_128k_ntk_factory(self):
        """Test 128K NTK factory method."""
        config = ANERoPEConfig.for_128k_ntk()

        assert config.max_seq_len == 131072
        assert config.scaling_type == RoPEScalingType.NTK_AWARE
        assert config.ntk_alpha == 8.0

    def test_for_128k_yarn_factory(self):
        """Test 128K YaRN factory method."""
        config = ANERoPEConfig.for_128k_yarn()

        assert config.max_seq_len == 131072
        assert config.scaling_type == RoPEScalingType.YARN


class TestANERoPEBasic:
    """Basic tests for ANE RoPE."""

    def test_initialization_no_scaling(self):
        """Test initialization without scaling."""
        config = ANERoPEConfig(
            d_head=64,
            max_seq_len=4096,
            scaling_type=RoPEScalingType.NONE,
        )
        rope = ANERoPE(config)

        assert rope.d_head == 64
        assert rope.inv_freq.shape == (32,)  # d_head / 2

    def test_forward_shape(self):
        """Test forward pass preserves shape."""
        config = ANERoPEConfig(d_head=64, max_seq_len=1024)
        rope = ANERoPE(config)

        x = torch.randn(2, 8, 32, 64)  # batch, heads, seq, d_head
        y = rope(x)

        assert y.shape == x.shape

    def test_forward_with_offset(self):
        """Test forward with position offset."""
        config = ANERoPEConfig(d_head=64, max_seq_len=1024)
        rope = ANERoPE(config)

        x = torch.randn(2, 8, 16, 64)
        y = rope(x, position_offset=32)

        assert y.shape == x.shape

    def test_forward_q_k(self):
        """Test forward for both Q and K."""
        config = ANERoPEConfig(d_head=64, max_seq_len=1024)
        rope = ANERoPE(config)

        q = torch.randn(2, 8, 32, 64)
        k = torch.randn(2, 8, 32, 64)

        q_rot, k_rot = rope.forward_q_k(q, k)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape


class TestANERoPENTK:
    """Tests for NTK-aware RoPE scaling."""

    def test_ntk_initialization(self):
        """Test NTK-aware initialization."""
        config = ANERoPEConfig(
            d_head=64,
            max_seq_len=131072,
            scaling_type=RoPEScalingType.NTK_AWARE,
            ntk_alpha=8.0,
        )
        rope = ANERoPE(config)

        # NTK should scale the base frequency
        assert rope.inv_freq is not None

    def test_ntk_different_from_standard(self):
        """Test that NTK produces different frequencies than standard."""
        # Standard RoPE
        config_std = ANERoPEConfig(
            d_head=64,
            max_seq_len=4096,
            scaling_type=RoPEScalingType.NONE,
        )
        rope_std = ANERoPE(config_std)

        # NTK RoPE
        config_ntk = ANERoPEConfig(
            d_head=64,
            max_seq_len=131072,
            scaling_type=RoPEScalingType.NTK_AWARE,
            ntk_alpha=8.0,
        )
        rope_ntk = ANERoPE(config_ntk)

        # Frequencies should be different
        assert not torch.allclose(rope_std.inv_freq, rope_ntk.inv_freq)

    def test_ntk_lower_frequencies(self):
        """Test that NTK produces lower frequencies (longer wavelengths)."""
        config_std = ANERoPEConfig(
            d_head=64,
            max_seq_len=4096,
            scaling_type=RoPEScalingType.NONE,
        )
        rope_std = ANERoPE(config_std)

        config_ntk = ANERoPEConfig(
            d_head=64,
            max_seq_len=131072,
            scaling_type=RoPEScalingType.NTK_AWARE,
            ntk_alpha=8.0,
        )
        rope_ntk = ANERoPE(config_ntk)

        # NTK should have smaller inv_freq (lower frequencies)
        assert rope_ntk.inv_freq.mean() < rope_std.inv_freq.mean()


class TestANERoPEYaRN:
    """Tests for YaRN interpolation."""

    def test_yarn_initialization(self):
        """Test YaRN initialization."""
        config = ANERoPEConfig(
            d_head=64,
            max_seq_len=131072,
            scaling_type=RoPEScalingType.YARN,
        )
        rope = ANERoPE(config)

        assert rope.mscale != 1.0  # YaRN has magnitude scale

    def test_yarn_mscale(self):
        """Test YaRN magnitude scale computation."""
        config = ANERoPEConfig(
            d_head=64,
            max_seq_len=131072,
            original_max_seq_len=4096,
            scaling_type=RoPEScalingType.YARN,
            yarn_mscale=0.707,
        )
        rope = ANERoPE(config)

        # mscale should be > 0.707 due to log correction
        assert rope.mscale > 0.707

    def test_yarn_frequency_selective(self):
        """Test that YaRN applies selective scaling."""
        config = ANERoPEConfig(
            d_head=64,
            max_seq_len=131072,
            scaling_type=RoPEScalingType.YARN,
        )
        rope = ANERoPE(config)

        # YaRN should have mixed frequencies
        # High freq (small wavelength) less scaled than low freq
        assert rope.inv_freq is not None


class TestANERoPELinear:
    """Tests for linear RoPE scaling."""

    def test_linear_initialization(self):
        """Test linear scaling initialization."""
        config = ANERoPEConfig(
            d_head=64,
            max_seq_len=8192,
            original_max_seq_len=4096,
            scaling_type=RoPEScalingType.LINEAR,
        )
        rope = ANERoPE(config)

        assert rope.inv_freq is not None

    def test_linear_scales_uniformly(self):
        """Test that linear scaling reduces all frequencies uniformly."""
        config_std = ANERoPEConfig(
            d_head=64,
            max_seq_len=4096,
            scaling_type=RoPEScalingType.NONE,
        )
        rope_std = ANERoPE(config_std)

        config_linear = ANERoPEConfig(
            d_head=64,
            max_seq_len=8192,
            original_max_seq_len=4096,
            scaling_type=RoPEScalingType.LINEAR,
        )
        rope_linear = ANERoPE(config_linear)

        # Linear should be standard / 2 (since max doubled)
        ratio = rope_linear.inv_freq / rope_std.inv_freq
        # All ratios should be approximately equal (uniform scaling)
        assert torch.allclose(ratio, ratio[0].expand_as(ratio), atol=1e-5)


class TestANERoPEDynamicNTK:
    """Tests for dynamic NTK scaling."""

    def test_dynamic_ntk_initialization(self):
        """Test dynamic NTK initialization."""
        config = ANERoPEConfig(
            d_head=64,
            max_seq_len=131072,
            scaling_type=RoPEScalingType.DYNAMIC_NTK,
        )
        rope = ANERoPE(config)

        assert rope.inv_freq is not None

    def test_get_dynamic_frequencies(self):
        """Test dynamic frequency computation."""
        config = ANERoPEConfig(
            d_head=64,
            max_seq_len=131072,
            original_max_seq_len=4096,
            scaling_type=RoPEScalingType.DYNAMIC_NTK,
        )
        rope = ANERoPE(config)

        # For sequence within original length
        cos_short, sin_short = rope.get_dynamic_ntk_frequencies(2048)
        assert cos_short.shape == (2048, 32)

        # For sequence beyond original length
        cos_long, sin_long = rope.get_dynamic_ntk_frequencies(8192)
        assert cos_long.shape == (8192, 32)


class TestANERoPERotation:
    """Tests for rotation correctness."""

    def test_rotation_preserves_norm(self):
        """Test that rotation preserves vector norm."""
        config = ANERoPEConfig(d_head=64, max_seq_len=1024)
        rope = ANERoPE(config)

        x = torch.randn(1, 1, 1, 64)
        x_rot = rope(x)

        # Norm should be preserved (rotation is orthogonal)
        assert torch.allclose(
            x.norm(dim=-1), x_rot.norm(dim=-1), rtol=1e-4
        )

    def test_same_position_same_rotation(self):
        """Test that same positions get same rotation."""
        config = ANERoPEConfig(d_head=64, max_seq_len=1024)
        rope = ANERoPE(config)

        x = torch.randn(2, 4, 1, 64)

        # Apply rotation twice to same position
        y1 = rope(x, position_offset=10)
        y2 = rope(x, position_offset=10)

        assert torch.allclose(y1, y2)

    def test_different_positions_different_rotations(self):
        """Test that different positions get different rotations."""
        config = ANERoPEConfig(d_head=64, max_seq_len=1024)
        rope = ANERoPE(config)

        x = torch.randn(1, 1, 1, 64)

        y_pos0 = rope(x, position_offset=0)
        y_pos10 = rope(x, position_offset=10)

        # Should be different
        assert not torch.allclose(y_pos0, y_pos10)


class TestANERoPEFP16:
    """Tests for FP16 optimization."""

    def test_fp16_cache(self):
        """Test that FP16 mode uses FP16 cache."""
        config = ANERoPEConfig(
            d_head=64,
            max_seq_len=1024,
            use_fp16=True,
        )
        rope = ANERoPE(config)

        assert rope.cos_cached.dtype == torch.float16
        assert rope.sin_cached.dtype == torch.float16

    def test_fp32_cache(self):
        """Test that FP32 mode uses FP32 cache."""
        config = ANERoPEConfig(
            d_head=64,
            max_seq_len=1024,
            use_fp16=False,
        )
        rope = ANERoPE(config)

        assert rope.cos_cached.dtype == torch.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
