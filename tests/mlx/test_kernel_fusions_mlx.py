"""
Tests for MLX Kernel Fusions

Tests cover:
- Metal feature detection
- SIMD-group reductions for softmax
- Fused RMSNorm
- Threadgroup memory attention patterns
- FP16/BF16 tiled matrix multiplication
"""

import pytest

# Check MLX availability
try:
    import mlx.core as mx
    import mlx.nn as nn

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None
    nn = None

pytestmark = pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")


# Only import if MLX is available to avoid import errors
if MLX_AVAILABLE:
    from deepseek.mlx.kernel_fusions import (
        AppleSiliconGen,
        MetalFeatures,
        MLXKernelDispatcher,
        RMSNormFused,
        TileConfig,
        attention_fused,
        attention_tiled,
        get_dispatcher,
        get_metal_features,
        matmul_bf16_tiled,
        matmul_fp16_tiled,
        rms_norm_fused,
        softmax_fused,
        softmax_fused_safe,
        swiglu_fused,
    )


# =============================================================================
# Metal Feature Detection Tests
# =============================================================================


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestMetalFeatures:
    """Tests for Metal feature detection."""

    def test_feature_detection(self) -> None:
        """Test basic feature detection."""
        features = get_metal_features()

        assert features.simd_width == 32
        assert features.max_threadgroup_size == 1024
        assert features.has_simd_group_reduce
        assert features.has_fp16

    def test_simd_group_reduce(self) -> None:
        """Test SIMD-group reduce availability."""
        features = MetalFeatures.detect()
        assert features.has_simd_group_reduce

    def test_generation_detection(self) -> None:
        """Test Apple Silicon generation detection."""
        features = get_metal_features()

        # Should be one of the valid generations
        assert features.generation in list(AppleSiliconGen)


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestAppleSiliconGen:
    """Tests for Apple Silicon generation enum."""

    def test_generation_values(self) -> None:
        """Test generation enum values."""
        assert AppleSiliconGen.M1.value == "m1"
        assert AppleSiliconGen.M2.value == "m2"
        assert AppleSiliconGen.M3.value == "m3"
        assert AppleSiliconGen.M4.value == "m4"


# =============================================================================
# Tile Configuration Tests
# =============================================================================


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestTileConfig:
    """Tests for tile configuration."""

    def test_default_config(self) -> None:
        """Test default tile configuration."""
        config = TileConfig()
        assert config.tile_m == 32
        assert config.tile_n == 32
        assert config.tile_k == 32

    def test_matmul_config_small(self) -> None:
        """Test tile config for small matrices."""
        config = TileConfig.for_matmul(m=32, n=32, k=32)
        assert config.tile_m == 16  # Small matrices use smaller tiles

    def test_matmul_config_large(self) -> None:
        """Test tile config for large matrices."""
        config = TileConfig.for_matmul(m=1024, n=1024, k=1024)
        assert config.tile_m >= 32  # Large matrices use larger tiles

    def test_attention_config(self) -> None:
        """Test tile config for attention."""
        config = TileConfig.for_attention(seq_len=512, head_dim=64)
        assert config.tile_k == 64  # Should match head_dim


# =============================================================================
# Fused Softmax Tests
# =============================================================================


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestFusedSoftmax:
    """Tests for fused softmax with SIMD-group reductions."""

    def test_softmax_basic(self) -> None:
        """Test basic softmax computation."""
        x = mx.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        result = softmax_fused(x, axis=-1)

        # Check shape
        assert result.shape == x.shape

        # Check sum to 1
        sums = mx.sum(result, axis=-1)
        assert mx.allclose(sums, mx.ones(2))

    def test_softmax_safe(self) -> None:
        """Test numerically stable softmax."""
        # Large values that could overflow without stability
        x = mx.array([[100.0, 200.0, 300.0]])
        result = softmax_fused_safe(x, axis=-1)

        # Should handle large values
        assert not mx.any(mx.isnan(result))
        assert not mx.any(mx.isinf(result))

    def test_softmax_3d(self) -> None:
        """Test softmax on 3D tensor."""
        x = mx.random.normal((4, 8, 64))
        result = softmax_fused(x, axis=-1)

        assert result.shape == x.shape

        # Verify sum to 1 on last axis
        sums = mx.sum(result, axis=-1)
        assert mx.allclose(sums, mx.ones((4, 8)), atol=1e-5)

    def test_softmax_different_axis(self) -> None:
        """Test softmax on different axis."""
        x = mx.random.normal((4, 8))
        result = softmax_fused(x, axis=0)

        # Sum along axis 0 should be 1
        sums = mx.sum(result, axis=0)
        assert mx.allclose(sums, mx.ones(8), atol=1e-5)


# =============================================================================
# Fused RMSNorm Tests
# =============================================================================


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestFusedRMSNorm:
    """Tests for fused RMSNorm with SIMD-group reductions."""

    def test_rmsnorm_module(self) -> None:
        """Test RMSNorm module."""
        norm = RMSNormFused(dims=64)
        x = mx.random.normal((2, 16, 64))

        result = norm(x)
        assert result.shape == x.shape

    def test_rmsnorm_function(self) -> None:
        """Test functional RMSNorm."""
        x = mx.random.normal((2, 16, 64))
        weight = mx.ones(64)

        result = rms_norm_fused(x, weight, eps=1e-6)
        assert result.shape == x.shape

    def test_rmsnorm_normalization(self) -> None:
        """Test that RMSNorm actually normalizes."""
        x = mx.random.normal((4, 32, 128))
        weight = mx.ones(128)

        result = rms_norm_fused(x, weight, eps=1e-6)

        # RMS should be approximately 1
        rms = mx.sqrt(mx.mean(mx.square(result), axis=-1))
        assert mx.allclose(rms, mx.ones(rms.shape), atol=0.1)

    def test_rmsnorm_with_scale(self) -> None:
        """Test RMSNorm with non-unit scale."""
        x = mx.random.normal((2, 8, 32))
        weight = mx.full((32,), 2.0)

        result = rms_norm_fused(x, weight, eps=1e-6)

        # RMS should be approximately 2
        rms = mx.sqrt(mx.mean(mx.square(result), axis=-1))
        assert mx.mean(rms) > 1.5  # Scaled up


# =============================================================================
# Fused Attention Tests
# =============================================================================


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestFusedAttention:
    """Tests for fused attention with threadgroup memory patterns."""

    def test_attention_basic(self) -> None:
        """Test basic attention computation."""
        batch, heads, seq_len, head_dim = 2, 4, 32, 64
        q = mx.random.normal((batch, heads, seq_len, head_dim))
        k = mx.random.normal((batch, heads, seq_len, head_dim))
        v = mx.random.normal((batch, heads, seq_len, head_dim))

        result = attention_fused(q, k, v)
        assert result.shape == (batch, heads, seq_len, head_dim)

    def test_attention_tiled(self) -> None:
        """Test tiled attention."""
        batch, heads, seq_len, head_dim = 1, 2, 64, 32
        q = mx.random.normal((batch, heads, seq_len, head_dim))
        k = mx.random.normal((batch, heads, seq_len, head_dim))
        v = mx.random.normal((batch, heads, seq_len, head_dim))

        result = attention_tiled(q, k, v, block_size=32)
        assert result.shape == (batch, heads, seq_len, head_dim)

    def test_attention_scale(self) -> None:
        """Test attention with custom scale."""
        batch, heads, seq_len, head_dim = 2, 4, 32, 64
        q = mx.random.normal((batch, heads, seq_len, head_dim))
        k = mx.random.normal((batch, heads, seq_len, head_dim))
        v = mx.random.normal((batch, heads, seq_len, head_dim))

        result = attention_fused(q, k, v, scale=0.1)
        assert result.shape == (batch, heads, seq_len, head_dim)

    def test_attention_with_mask(self) -> None:
        """Test attention with mask."""
        batch, heads, seq_len, head_dim = 1, 2, 16, 32
        q = mx.random.normal((batch, heads, seq_len, head_dim))
        k = mx.random.normal((batch, heads, seq_len, head_dim))
        v = mx.random.normal((batch, heads, seq_len, head_dim))

        # Causal mask
        mask = mx.triu(mx.full((seq_len, seq_len), -float("inf")), k=1)

        result = attention_fused(q, k, v, mask=mask)
        assert result.shape == (batch, heads, seq_len, head_dim)


# =============================================================================
# FP16/BF16 Matrix Multiply Tests
# =============================================================================


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestMatmulTiled:
    """Tests for tiled matrix multiplication."""

    def test_matmul_fp16(self) -> None:
        """Test FP16 matrix multiply."""
        a = mx.random.normal((64, 128))
        b = mx.random.normal((128, 64))

        result = matmul_fp16_tiled(a, b)

        # Verify shape
        assert result.shape == (64, 64)

        # Verify dtype
        assert result.dtype == mx.float16

    def test_matmul_bf16(self) -> None:
        """Test BF16 matrix multiply (if supported)."""
        features = get_metal_features()

        a = mx.random.normal((32, 64))
        b = mx.random.normal((64, 32))

        result = matmul_bf16_tiled(a, b)
        assert result.shape == (32, 32)

        # Dtype depends on BF16 support
        if features.has_bf16:
            assert result.dtype == mx.bfloat16
        else:
            assert result.dtype == mx.float16

    def test_matmul_accuracy(self) -> None:
        """Test matmul accuracy."""
        a = mx.random.normal((32, 64))
        b = mx.random.normal((64, 32))

        result = matmul_fp16_tiled(a.astype(mx.float32), b.astype(mx.float32))
        expected = mx.matmul(a.astype(mx.float16), b.astype(mx.float16))

        # Should be close
        diff = mx.abs(result.astype(mx.float32) - expected.astype(mx.float32))
        assert mx.mean(diff) < 0.1


# =============================================================================
# SwiGLU Tests
# =============================================================================


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestFusedSwiGLU:
    """Tests for fused SwiGLU activation."""

    def test_swiglu_basic(self) -> None:
        """Test basic SwiGLU computation."""
        gate = mx.random.normal((4, 32, 128))
        up = mx.random.normal((4, 32, 128))

        result = swiglu_fused(gate, up)
        assert result.shape == gate.shape

    def test_swiglu_correctness(self) -> None:
        """Test SwiGLU correctness."""
        gate = mx.array([[0.0, 1.0, -1.0]])
        up = mx.array([[1.0, 1.0, 1.0]])

        result = swiglu_fused(gate, up)

        # SwiGLU(gate, up) = silu(gate) * up
        expected = nn.silu(gate) * up
        assert mx.allclose(result, expected, atol=1e-5)


# =============================================================================
# Kernel Dispatcher Tests
# =============================================================================


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestMLXKernelDispatcher:
    """Tests for MLX kernel dispatcher."""

    def test_dispatcher_initialization(self) -> None:
        """Test dispatcher initialization."""
        dispatcher = get_dispatcher()
        assert dispatcher is not None
        assert dispatcher.features is not None

    def test_dispatcher_softmax(self) -> None:
        """Test dispatcher softmax."""
        dispatcher = MLXKernelDispatcher()
        x = mx.random.normal((4, 64))

        result = dispatcher.softmax(x)
        assert result.shape == x.shape

    def test_dispatcher_rmsnorm(self) -> None:
        """Test dispatcher RMSNorm."""
        dispatcher = MLXKernelDispatcher()
        x = mx.random.normal((2, 16, 64))
        weight = mx.ones(64)

        result = dispatcher.rms_norm(x, weight)
        assert result.shape == x.shape

    def test_dispatcher_attention(self) -> None:
        """Test dispatcher attention."""
        dispatcher = MLXKernelDispatcher()
        batch, heads, seq_len, head_dim = 1, 2, 16, 32
        q = mx.random.normal((batch, heads, seq_len, head_dim))
        k = mx.random.normal((batch, heads, seq_len, head_dim))
        v = mx.random.normal((batch, heads, seq_len, head_dim))

        result = dispatcher.attention(q, k, v)
        assert result.shape == (batch, heads, seq_len, head_dim)

    def test_dispatcher_matmul(self) -> None:
        """Test dispatcher matmul."""
        dispatcher = MLXKernelDispatcher()
        a = mx.random.normal((32, 64))
        b = mx.random.normal((64, 32))

        result = dispatcher.matmul(a, b)
        assert result.shape == (32, 32)

    def test_dispatcher_swiglu(self) -> None:
        """Test dispatcher SwiGLU."""
        dispatcher = MLXKernelDispatcher()
        gate = mx.random.normal((2, 16, 64))
        up = mx.random.normal((2, 16, 64))

        result = dispatcher.swiglu(gate, up)
        assert result.shape == gate.shape


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestMLXKernelIntegration:
    """Integration tests for MLX kernel fusions."""

    def test_transformer_block_ops(self) -> None:
        """Test operations similar to a transformer block."""
        batch, seq_len, hidden_size = 2, 32, 256
        num_heads = 4
        head_dim = hidden_size // num_heads

        # Input
        x = mx.random.normal((batch, seq_len, hidden_size))

        # RMSNorm
        norm_weight = mx.ones(hidden_size)
        x_norm = rms_norm_fused(x, norm_weight)
        assert x_norm.shape == x.shape

        # Reshape for attention
        x_reshaped = x_norm.reshape(batch, seq_len, num_heads, head_dim)
        q = mx.transpose(x_reshaped, (0, 2, 1, 3))
        k = mx.transpose(x_reshaped, (0, 2, 1, 3))
        v = mx.transpose(x_reshaped, (0, 2, 1, 3))

        # Attention
        attn_out = attention_fused(q, k, v)
        assert attn_out.shape == (batch, num_heads, seq_len, head_dim)

        # Reshape back
        out = mx.transpose(attn_out, (0, 2, 1, 3))
        out = out.reshape(batch, seq_len, hidden_size)
        assert out.shape == x.shape

    def test_mlp_block_ops(self) -> None:
        """Test operations similar to an MLP block."""
        batch, seq_len, hidden_size = 2, 32, 256
        intermediate_size = hidden_size * 4

        # Input (hidden_size is used for context)
        _ = hidden_size  # Input would be projected to intermediate_size

        # Gate and up projections (simulated)
        gate = mx.random.normal((batch, seq_len, intermediate_size))
        up = mx.random.normal((batch, seq_len, intermediate_size))

        # Fused SwiGLU
        activated = swiglu_fused(gate, up)
        assert activated.shape == (batch, seq_len, intermediate_size)

        # Down projection would go here

    def test_kernel_fusion_pipeline(self) -> None:
        """Test a pipeline of fused kernel operations."""
        x = mx.random.normal((4, 64, 128))
        weight = mx.ones(128)

        # Chain of operations
        x = rms_norm_fused(x, weight)
        x = softmax_fused(x, axis=-1)

        # Verify final shape and values
        assert x.shape == (4, 64, 128)
        sums = mx.sum(x, axis=-1)
        assert mx.allclose(sums, mx.ones((4, 64)), atol=1e-5)
