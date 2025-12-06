"""
Tests for PyTorch CUDA Features and Fused Kernels

Tests cover:
- CUDA feature detection (TMA, Warpgroup, FP8)
- Fused softmax with SIMD reductions
- Fused RMSNorm
- Fused attention with threadgroup memory patterns
- FP16 tiled matrix multiplication
"""

import pytest
import torch
import torch.nn.functional as F

from deepseek.torch.kernels.cuda_features import (
    ComputeCapability,
    CUDAFeatures,
    FP8Config,
    FP8Format,
    KernelBackend,
    TMADescriptor,
    WarpgroupConfig,
    fp8_matmul,
    select_backend,
)
from deepseek.torch.kernels.fused_kernels import (
    TRITON_AVAILABLE,
    FusedKernelDispatcher,
    MatmulConfig,
    attention_fused,
    get_dispatcher,
    matmul_fp16_tiled,
    rms_norm_fused,
    softmax_fused,
)

# =============================================================================
# CUDA Feature Detection Tests
# =============================================================================


class TestComputeCapability:
    """Tests for ComputeCapability dataclass."""

    def test_at_least(self) -> None:
        """Test at_least method."""
        cap = ComputeCapability(major=8, minor=0)
        assert cap.at_least(8, 0)
        assert cap.at_least(7, 5)
        assert not cap.at_least(9, 0)

    def test_generation_detection(self) -> None:
        """Test GPU generation detection."""
        # Hopper
        hopper = ComputeCapability(major=9, minor=0)
        assert hopper.is_hopper()
        assert hopper.is_ampere()  # is_ampere means >= 8.0
        assert hopper.is_turing()

        # Ampere
        ampere = ComputeCapability(major=8, minor=0)
        assert not ampere.is_hopper()
        assert ampere.is_ampere()
        assert ampere.is_turing()

        # Turing
        turing = ComputeCapability(major=7, minor=5)
        assert not turing.is_hopper()
        assert not turing.is_ampere()
        assert turing.is_turing()

    def test_str_representation(self) -> None:
        """Test string representation."""
        cap = ComputeCapability(major=9, minor=0)
        assert str(cap) == "SM 9.0"


class TestCUDAFeatures:
    """Tests for CUDA feature detection."""

    def test_feature_detection(self) -> None:
        """Test basic feature detection."""
        # This works even without CUDA
        features = CUDAFeatures.detect()

        # Basic validation
        assert isinstance(features.has_tma, bool)
        assert isinstance(features.has_warpgroup, bool)
        assert isinstance(features.has_fp8_tensor_core, bool)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_feature_detection(self) -> None:
        """Test feature detection with actual CUDA device."""
        device = torch.device("cuda:0")
        features = CUDAFeatures.detect(device)

        # Should have detected something
        assert features.compute_capability is not None
        assert features.compute_capability.major >= 0

    def test_default_features_without_cuda(self) -> None:
        """Test default features on CPU."""
        features = CUDAFeatures.detect(torch.device("cpu"))

        # Should return conservative defaults
        assert features.compute_capability.major == 0
        assert not features.has_tma
        assert not features.has_warpgroup

    def test_h100_preset(self) -> None:
        """Test H100 preset features."""
        features = CUDAFeatures.h100()
        assert features.has_tma
        assert features.has_warpgroup
        assert features.has_fp8_tensor_core

    def test_a100_preset(self) -> None:
        """Test A100 preset features."""
        features = CUDAFeatures.a100()
        assert not features.has_tma
        assert not features.has_warpgroup
        assert features.has_bf16_tensor_core

    def test_optimal_tile_size(self) -> None:
        """Test optimal tile size from features."""
        h100 = CUDAFeatures.h100()
        tile_m, tile_n, tile_k = h100.optimal_tile_size()
        assert tile_m >= 128  # Hopper should have large tiles

        a100 = CUDAFeatures.a100()
        tile_m, tile_n, tile_k = a100.optimal_tile_size()
        assert tile_m >= 64  # Ampere should have reasonable tiles


class TestKernelBackend:
    """Tests for kernel backend selection."""

    def test_backend_selection(self) -> None:
        """Test backend selection based on device."""
        # CPU should get CPU backend
        backend = select_backend(torch.device("cpu"))
        assert backend == KernelBackend.CPU

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_backend_selection(self) -> None:
        """Test backend selection for CUDA device."""
        backend = select_backend(torch.device("cuda:0"))
        assert backend in [
            KernelBackend.CPU,
            KernelBackend.CUDA_STANDARD,
            KernelBackend.CUDA_AMPERE,
            KernelBackend.CUDA_HOPPER,
        ]


# =============================================================================
# Fused Kernel Tests
# =============================================================================


class TestFusedSoftmax:
    """Tests for fused softmax operation."""

    def test_softmax_cpu(self) -> None:
        """Test softmax on CPU."""
        x = torch.randn(4, 64)
        result = softmax_fused(x, dim=-1)

        # Compare with PyTorch
        expected = F.softmax(x, dim=-1)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_softmax_2d(self) -> None:
        """Test softmax on 2D tensor."""
        x = torch.randn(32, 128)
        result = softmax_fused(x, dim=-1)

        # Verify properties of softmax
        assert torch.allclose(result.sum(dim=-1), torch.ones(32), atol=1e-5)
        assert (result >= 0).all()
        assert (result <= 1).all()

    def test_softmax_3d(self) -> None:
        """Test softmax on 3D tensor."""
        x = torch.randn(4, 8, 64)
        result = softmax_fused(x, dim=-1)

        expected = F.softmax(x, dim=-1)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_softmax_cuda(self) -> None:
        """Test softmax on CUDA."""
        x = torch.randn(32, 128, device="cuda")
        result = softmax_fused(x, dim=-1)

        expected = F.softmax(x, dim=-1)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)


class TestFusedRMSNorm:
    """Tests for fused RMSNorm operation."""

    def test_rmsnorm_cpu(self) -> None:
        """Test RMSNorm on CPU."""
        hidden_size = 128
        x = torch.randn(4, 32, hidden_size)
        weight = torch.ones(hidden_size)

        result = rms_norm_fused(x, weight, eps=1e-6)

        # Verify output shape
        assert result.shape == x.shape

        # Verify approximate normalization
        variance = result.pow(2).mean(-1)
        assert torch.allclose(variance, torch.ones_like(variance), atol=0.1)

    def test_rmsnorm_with_scale(self) -> None:
        """Test RMSNorm with learnable scale."""
        hidden_size = 64
        x = torch.randn(2, 16, hidden_size)
        weight = torch.full((hidden_size,), 2.0)

        result = rms_norm_fused(x, weight, eps=1e-6)

        # Scale should affect magnitude
        variance = result.pow(2).mean(-1)
        assert variance.mean() > 1.0  # Should be scaled up

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_rmsnorm_cuda(self) -> None:
        """Test RMSNorm on CUDA."""
        hidden_size = 256
        x = torch.randn(8, 64, hidden_size, device="cuda")
        weight = torch.ones(hidden_size, device="cuda")

        result = rms_norm_fused(x, weight, eps=1e-6)
        assert result.shape == x.shape


class TestFusedAttention:
    """Tests for fused attention operation."""

    def test_attention_cpu(self) -> None:
        """Test attention on CPU."""
        batch, heads, seq_len, head_dim = 2, 4, 32, 64
        q = torch.randn(batch, heads, seq_len, head_dim)
        k = torch.randn(batch, heads, seq_len, head_dim)
        v = torch.randn(batch, heads, seq_len, head_dim)

        result = attention_fused(q, k, v)

        # Verify output shape
        assert result.shape == (batch, heads, seq_len, head_dim)

    def test_attention_causal(self) -> None:
        """Test causal attention."""
        batch, heads, seq_len, head_dim = 1, 2, 16, 32
        q = torch.randn(batch, heads, seq_len, head_dim)
        k = torch.randn(batch, heads, seq_len, head_dim)
        v = torch.randn(batch, heads, seq_len, head_dim)

        result = attention_fused(q, k, v, causal=True)
        assert result.shape == (batch, heads, seq_len, head_dim)

    def test_attention_scale(self) -> None:
        """Test attention with custom scale."""
        batch, heads, seq_len, head_dim = 2, 4, 32, 64
        q = torch.randn(batch, heads, seq_len, head_dim)
        k = torch.randn(batch, heads, seq_len, head_dim)
        v = torch.randn(batch, heads, seq_len, head_dim)

        result = attention_fused(q, k, v, scale=0.1)
        assert result.shape == (batch, heads, seq_len, head_dim)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_attention_cuda(self) -> None:
        """Test attention on CUDA."""
        batch, heads, seq_len, head_dim = 2, 8, 128, 64
        q = torch.randn(batch, heads, seq_len, head_dim, device="cuda")
        k = torch.randn(batch, heads, seq_len, head_dim, device="cuda")
        v = torch.randn(batch, heads, seq_len, head_dim, device="cuda")

        result = attention_fused(q, k, v)
        assert result.shape == (batch, heads, seq_len, head_dim)
        assert result.device.type == "cuda"


class TestMatmulTiled:
    """Tests for tiled matrix multiplication."""

    def test_matmul_cpu(self) -> None:
        """Test matmul on CPU."""
        a = torch.randn(64, 128)
        b = torch.randn(128, 64)

        result = matmul_fp16_tiled(a, b)
        expected = torch.matmul(a, b)

        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)

    def test_matmul_config(self) -> None:
        """Test matmul with custom config."""
        config = MatmulConfig(block_m=64, block_n=64, block_k=16)
        a = torch.randn(128, 256)
        b = torch.randn(256, 128)

        result = matmul_fp16_tiled(a, b, config)
        expected = torch.matmul(a, b)

        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_matmul_cuda(self) -> None:
        """Test matmul on CUDA."""
        a = torch.randn(128, 256, device="cuda", dtype=torch.float16)
        b = torch.randn(256, 128, device="cuda", dtype=torch.float16)

        result = matmul_fp16_tiled(a, b)
        expected = torch.matmul(a, b)

        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)


# =============================================================================
# Kernel Dispatcher Tests
# =============================================================================


class TestFusedKernelDispatcher:
    """Tests for the kernel dispatcher."""

    def test_dispatcher_initialization(self) -> None:
        """Test dispatcher initialization."""
        dispatcher = get_dispatcher()
        assert dispatcher is not None
        assert dispatcher.backend is not None

    def test_dispatcher_softmax(self) -> None:
        """Test dispatcher softmax."""
        dispatcher = FusedKernelDispatcher()
        x = torch.randn(4, 64)

        result = dispatcher.softmax(x)
        expected = F.softmax(x, dim=-1)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_dispatcher_rmsnorm(self) -> None:
        """Test dispatcher RMSNorm."""
        dispatcher = FusedKernelDispatcher()
        x = torch.randn(2, 16, 64)
        weight = torch.ones(64)

        result = dispatcher.rms_norm(x, weight)
        assert result.shape == x.shape

    def test_dispatcher_attention(self) -> None:
        """Test dispatcher attention."""
        dispatcher = FusedKernelDispatcher()
        batch, heads, seq_len, head_dim = 1, 2, 16, 32
        q = torch.randn(batch, heads, seq_len, head_dim)
        k = torch.randn(batch, heads, seq_len, head_dim)
        v = torch.randn(batch, heads, seq_len, head_dim)

        result = dispatcher.attention(q, k, v)
        assert result.shape == (batch, heads, seq_len, head_dim)

    def test_dispatcher_matmul(self) -> None:
        """Test dispatcher matmul."""
        dispatcher = FusedKernelDispatcher()
        a = torch.randn(32, 64)
        b = torch.randn(64, 32)

        result = dispatcher.matmul(a, b)
        expected = torch.matmul(a, b)
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)


# =============================================================================
# FP8 Tests
# =============================================================================


class TestFP8Config:
    """Tests for FP8 configuration."""

    def test_fp8_formats(self) -> None:
        """Test FP8 format enum."""
        assert FP8Format.E4M3 != FP8Format.E5M2

    def test_fp8_config(self) -> None:
        """Test FP8 config creation."""
        config = FP8Config(
            format=FP8Format.E4M3,
            scale=1.0,
        )
        assert config.format == FP8Format.E4M3
        assert config.scale == 1.0

    def test_fp8_matmul_fallback(self) -> None:
        """Test FP8 matmul falls back correctly."""
        a = torch.randn(32, 64)
        b = torch.randn(64, 32)

        result = fp8_matmul(a, b)
        expected = torch.matmul(a, b)

        # Should match since FP8 is not available
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)


# =============================================================================
# TMA and Warpgroup Tests
# =============================================================================


class TestTMADescriptor:
    """Tests for TMA descriptor."""

    def test_tma_descriptor_creation(self) -> None:
        """Test TMA descriptor creation."""
        tensor = torch.randn(256, 256)
        desc = TMADescriptor(
            tensor=tensor,
            tile_shape=(64, 64),
            swizzle_mode="128B",
        )
        assert desc.tile_shape == (64, 64)
        assert desc.swizzle_mode == "128B"


class TestWarpgroupConfig:
    """Tests for warpgroup configuration."""

    def test_warpgroup_constants(self) -> None:
        """Test warpgroup constant values."""
        assert WarpgroupConfig.WARP_SIZE == 32
        assert WarpgroupConfig.NUM_WARPS == 4
        assert WarpgroupConfig.WARPGROUP_SIZE == 128

    def test_warpgroup_count(self) -> None:
        """Test warpgroup count calculation."""
        assert WarpgroupConfig.get_num_warpgroups(32) == 1
        assert WarpgroupConfig.get_num_warpgroups(128) == 1
        assert WarpgroupConfig.get_num_warpgroups(256) == 2


# =============================================================================
# Integration Tests
# =============================================================================


class TestKernelIntegration:
    """Integration tests for kernel fusions."""

    def test_full_transformer_block_ops(self) -> None:
        """Test operations similar to a transformer block."""
        batch, seq_len, hidden_size = 2, 32, 256
        num_heads = 4
        head_dim = hidden_size // num_heads

        # Input
        x = torch.randn(batch, seq_len, hidden_size)

        # RMSNorm
        norm_weight = torch.ones(hidden_size)
        x_norm = rms_norm_fused(x, norm_weight)
        assert x_norm.shape == x.shape

        # Project to Q, K, V
        q = x_norm.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
        k = x_norm.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
        v = x_norm.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)

        # Attention
        attn_out = attention_fused(q, k, v)
        assert attn_out.shape == (batch, num_heads, seq_len, head_dim)

        # Reshape back
        out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, hidden_size)
        assert out.shape == x.shape

    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
    def test_triton_kernel_compilation(self) -> None:
        """Test that Triton kernels compile correctly."""
        # This test verifies kernel compilation
        x = torch.randn(32, 128, device="cuda")
        _ = softmax_fused(x, dim=-1)
        # If we get here, compilation succeeded
