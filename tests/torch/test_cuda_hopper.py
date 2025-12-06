"""
Tests for CUDA Hopper Features (TMA, Warpgroup, FP8)

These tests verify the CUDA Hopper feature implementations including:
- TMA (Tensor Memory Accelerator) descriptors and operations
- Warpgroup collective operations (128 threads)
- FP8 tensor core support
- HopperKernelDispatcher

Tests use CPU fallbacks when CUDA is not available.
"""

import pytest
import torch

from deepseek.torch.kernels.cuda_features import (
    ComputeCapability,
    CUDAFeatures,
    FP8Config,
    FP8Format,
    HopperKernelDispatcher,
    KernelBackend,
    KernelStats,
    TMADescriptor,
    WarpgroupConfig,
    create_tma_descriptor,
    fp8_matmul,
    get_hopper_dispatcher,
    get_kernel_stats,
    reset_kernel_stats,
    select_backend,
    tma_async_copy_global_to_shared,
    tma_prefetch,
    warpgroup_matmul,
    warpgroup_reduce_max,
    warpgroup_reduce_sum,
)


# =============================================================================
# Compute Capability Tests
# =============================================================================


class TestComputeCapability:
    """Tests for ComputeCapability dataclass."""

    def test_creation(self) -> None:
        """Test creating compute capability."""
        cc = ComputeCapability(major=9, minor=0)
        assert cc.major == 9
        assert cc.minor == 0

    def test_at_least(self) -> None:
        """Test at_least method."""
        hopper = ComputeCapability(major=9, minor=0)
        assert hopper.at_least(9, 0)
        assert hopper.at_least(8, 0)
        assert hopper.at_least(7, 5)
        assert not hopper.at_least(9, 1)
        assert not hopper.at_least(10, 0)

    def test_generation_detection(self) -> None:
        """Test GPU generation detection methods."""
        hopper = ComputeCapability(major=9, minor=0)
        assert hopper.is_hopper()
        assert hopper.is_ampere()
        assert hopper.is_turing()
        assert hopper.is_volta()

        ampere = ComputeCapability(major=8, minor=0)
        assert not ampere.is_hopper()
        assert ampere.is_ampere()
        assert ampere.is_turing()
        assert ampere.is_volta()

        turing = ComputeCapability(major=7, minor=5)
        assert not turing.is_hopper()
        assert not turing.is_ampere()
        assert turing.is_turing()
        assert turing.is_volta()

    def test_str_representation(self) -> None:
        """Test string representation."""
        cc = ComputeCapability(major=9, minor=0)
        assert str(cc) == "SM 9.0"


# =============================================================================
# CUDA Features Tests
# =============================================================================


class TestCUDAFeatures:
    """Tests for CUDA feature detection."""

    def test_default_features(self) -> None:
        """Test default feature values."""
        features = CUDAFeatures()
        assert features.has_tma is False
        assert features.has_warpgroup is False
        assert features.has_fp8_tensor_core is False
        assert features.warp_size == 32
        assert features.max_threads_per_block == 1024

    def test_h100_preset(self) -> None:
        """Test H100 preset configuration."""
        features = CUDAFeatures.h100()
        
        assert features.has_tma
        assert features.has_warpgroup
        assert features.has_fp8_tensor_core
        assert features.has_cluster
        assert features.has_bf16_tensor_core
        assert features.has_tf32
        assert features.has_async_copy
        assert features.compute_capability.is_hopper()
        assert features.sm_count == 132
        assert features.total_memory_gb == 80.0

    def test_a100_preset(self) -> None:
        """Test A100 preset configuration."""
        features = CUDAFeatures.a100()
        
        assert not features.has_tma
        assert not features.has_warpgroup
        assert not features.has_fp8_tensor_core
        assert features.has_bf16_tensor_core
        assert features.has_tf32
        assert features.has_async_copy
        assert features.compute_capability.is_ampere()
        assert features.sm_count == 108

    def test_detection_without_cuda(self) -> None:
        """Test feature detection without CUDA."""
        features = CUDAFeatures.detect()
        
        # Should not crash even without CUDA
        assert isinstance(features.has_tma, bool)
        assert isinstance(features.has_warpgroup, bool)

    def test_optimal_tile_size(self) -> None:
        """Test optimal tile size selection."""
        hopper = CUDAFeatures.h100()
        assert hopper.optimal_tile_size() == (256, 128, 64)
        
        ampere = CUDAFeatures.a100()
        assert ampere.optimal_tile_size() == (128, 128, 32)

    def test_optimal_block_size(self) -> None:
        """Test optimal block size calculation."""
        features = CUDAFeatures.h100()
        
        # Warpgroup-aligned (128 threads)
        assert features.optimal_block_size(200) == 256
        assert features.optimal_block_size(100) <= 256


# =============================================================================
# Kernel Backend Selection Tests
# =============================================================================


class TestKernelBackend:
    """Tests for kernel backend selection."""

    def test_select_cpu(self) -> None:
        """Test CPU backend selection."""
        device = torch.device("cpu")
        backend = select_backend(device)
        assert backend == KernelBackend.CPU

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_select_cuda(self) -> None:
        """Test CUDA backend selection."""
        device = torch.device("cuda:0")
        backend = select_backend(device)
        
        # Should be one of the CUDA backends
        assert backend in (
            KernelBackend.CUDA_HOPPER,
            KernelBackend.CUDA_AMPERE,
            KernelBackend.CUDA_STANDARD,
        )


# =============================================================================
# TMA Tests
# =============================================================================


class TestTMADescriptor:
    """Tests for TMA descriptor."""

    def test_descriptor_creation(self) -> None:
        """Test creating TMA descriptor."""
        tensor = torch.randn(128, 4096)
        desc = create_tma_descriptor(tensor, (32, 32))
        
        assert isinstance(desc, TMADescriptor)
        assert desc.tensor is tensor
        assert desc.tile_shape == (32, 32)

    def test_swizzle_modes(self) -> None:
        """Test different swizzle modes."""
        tensor = torch.randn(256, 256)
        
        # Different swizzle modes should all work
        for swizzle in ["none", "32B", "64B", "128B"]:
            desc = create_tma_descriptor(tensor, (32, 32), swizzle=swizzle)
            assert desc.swizzle_mode == swizzle


class TestTMAOperations:
    """Tests for TMA operations."""

    def test_async_copy_cpu_fallback(self) -> None:
        """Test async copy on CPU (should fallback gracefully)."""
        src = torch.randn(64, 64)
        result = tma_async_copy_global_to_shared(src, (16, 16))
        
        assert result.shape == src.shape
        # On CPU, should just be a copy
        assert torch.allclose(result, src)

    def test_prefetch_cpu_fallback(self) -> None:
        """Test prefetch on CPU (should be no-op)."""
        tensor = torch.randn(128, 128)
        
        # Should not raise any errors
        tma_prefetch(tensor, (0, 0))


# =============================================================================
# Warpgroup Tests
# =============================================================================


class TestWarpgroupConfig:
    """Tests for warpgroup configuration."""

    def test_default_config(self) -> None:
        """Test default warpgroup configuration."""
        config = WarpgroupConfig
        assert config.WARPGROUP_SIZE == 128
        assert config.WARP_SIZE == 32
        assert config.NUM_WARPS == 4

    def test_num_warpgroups(self) -> None:
        """Test calculating number of warpgroups."""
        assert WarpgroupConfig.get_num_warpgroups(128) == 1
        assert WarpgroupConfig.get_num_warpgroups(256) == 2
        assert WarpgroupConfig.get_num_warpgroups(384) == 3
        assert WarpgroupConfig.get_num_warpgroups(64) == 1  # Rounds up


class TestWarpgroupOperations:
    """Tests for warpgroup operations."""

    def test_warpgroup_matmul_cpu(self) -> None:
        """Test warpgroup matmul on CPU."""
        a = torch.randn(32, 64)
        b = torch.randn(64, 48)
        
        result = warpgroup_matmul(a, b)
        
        assert result.shape == (32, 48)
        # Should match standard matmul
        expected = torch.matmul(a, b)
        assert torch.allclose(result, expected)

    def test_warpgroup_reduce_sum(self) -> None:
        """Test warpgroup sum reduction."""
        tensor = torch.randn(4, 256)
        
        result = warpgroup_reduce_sum(tensor, dim=-1)
        expected = tensor.sum(dim=-1)
        
        assert torch.allclose(result, expected)

    def test_warpgroup_reduce_max(self) -> None:
        """Test warpgroup max reduction."""
        tensor = torch.randn(4, 256)
        
        result = warpgroup_reduce_max(tensor, dim=-1)
        expected = tensor.max(dim=-1)[0]
        
        assert torch.allclose(result, expected)


# =============================================================================
# FP8 Tests
# =============================================================================


class TestFP8Format:
    """Tests for FP8 format."""

    def test_format_values(self) -> None:
        """Test FP8 format enum values."""
        assert FP8Format.E4M3.value == "e4m3"
        assert FP8Format.E5M2.value == "e5m2"


class TestFP8Config:
    """Tests for FP8 configuration."""

    def test_default_config(self) -> None:
        """Test default FP8 configuration."""
        config = FP8Config()
        
        assert config.format == FP8Format.E4M3
        assert config.scale == 1.0
        assert config.amax_history_len == 16

    def test_to_fp8_fallback(self) -> None:
        """Test FP8 conversion fallback when not available."""
        config = FP8Config()
        tensor = torch.randn(16, 16)

        # On CPU, FP8 operations may not be fully supported
        # Test that it either works or raises expected error
        try:
            result = config.to_fp8(tensor)
            assert result.shape == tensor.shape
        except NotImplementedError:
            # CPU doesn't support FP8 mul - this is expected behavior
            pytest.skip("FP8 not supported on CPU")


class TestFP8Matmul:
    """Tests for FP8 matrix multiplication."""

    def test_fp8_matmul_cpu(self) -> None:
        """Test FP8 matmul on CPU (falls back to standard)."""
        a = torch.randn(16, 32)
        b = torch.randn(32, 24)
        
        result = fp8_matmul(a, b)
        
        assert result.shape == (16, 24)
        # On CPU, should match standard matmul
        expected = torch.matmul(a.float(), b.float())
        assert torch.allclose(result, expected, atol=1e-5)

    def test_fp8_matmul_with_config(self) -> None:
        """Test FP8 matmul with explicit config."""
        config = FP8Config(format=FP8Format.E5M2)
        a = torch.randn(8, 16)
        b = torch.randn(16, 12)
        
        result = fp8_matmul(a, b, config=config)
        
        assert result.shape == (8, 12)


# =============================================================================
# Hopper Kernel Dispatcher Tests
# =============================================================================


class TestHopperKernelDispatcher:
    """Tests for Hopper kernel dispatcher."""

    def test_initialization(self) -> None:
        """Test dispatcher initialization."""
        dispatcher = HopperKernelDispatcher()
        
        assert dispatcher.device is not None
        assert dispatcher.features is not None

    def test_matmul(self) -> None:
        """Test dispatcher matmul."""
        dispatcher = HopperKernelDispatcher()
        
        a = torch.randn(16, 32)
        b = torch.randn(32, 24)
        
        result = dispatcher.matmul(a, b)
        
        assert result.shape == (16, 24)

    def test_attention(self) -> None:
        """Test dispatcher attention."""
        dispatcher = HopperKernelDispatcher()
        
        q = torch.randn(1, 2, 32, 16)
        k = torch.randn(1, 2, 32, 16)
        v = torch.randn(1, 2, 32, 16)
        
        result = dispatcher.attention(q, k, v)
        
        assert result.shape == (1, 2, 32, 16)

    def test_softmax(self) -> None:
        """Test dispatcher softmax."""
        dispatcher = HopperKernelDispatcher()
        
        tensor = torch.randn(4, 256)
        result = dispatcher.softmax(tensor, dim=-1)
        
        assert result.shape == (4, 256)
        # Verify softmax sums to 1
        sums = result.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_rms_norm(self) -> None:
        """Test dispatcher RMS norm."""
        dispatcher = HopperKernelDispatcher()
        
        x = torch.randn(2, 128)
        weight = torch.ones(128)
        
        result = dispatcher.rms_norm(x, weight)
        
        assert result.shape == (2, 128)


class TestGlobalDispatcher:
    """Tests for global dispatcher singleton."""

    def test_get_hopper_dispatcher(self) -> None:
        """Test getting global dispatcher."""
        dispatcher = get_hopper_dispatcher()
        
        assert dispatcher is not None
        assert isinstance(dispatcher, HopperKernelDispatcher)

    def test_dispatcher_singleton(self) -> None:
        """Test that dispatcher is a singleton."""
        dispatcher1 = get_hopper_dispatcher()
        dispatcher2 = get_hopper_dispatcher()
        
        assert dispatcher1 is dispatcher2


# =============================================================================
# Kernel Statistics Tests
# =============================================================================


class TestKernelStats:
    """Tests for kernel statistics."""

    def test_record(self) -> None:
        """Test recording kernel invocation."""
        reset_kernel_stats()
        stats = get_kernel_stats()
        
        stats.record("test_kernel", time_ms=1.5)
        
        assert stats.total_invocations == 1
        assert stats.kernel_counts["test_kernel"] == 1
        assert stats.total_time_ms == 1.5

    def test_multiple_records(self) -> None:
        """Test recording multiple invocations."""
        reset_kernel_stats()
        stats = get_kernel_stats()
        
        stats.record("kernel_a", time_ms=1.0)
        stats.record("kernel_a", time_ms=2.0)
        stats.record("kernel_b", time_ms=0.5)
        
        assert stats.total_invocations == 3
        assert stats.kernel_counts["kernel_a"] == 2
        assert stats.kernel_counts["kernel_b"] == 1
        assert stats.total_time_ms == 3.5

    def test_reset(self) -> None:
        """Test resetting statistics."""
        stats = get_kernel_stats()
        stats.record("test", time_ms=1.0)
        
        reset_kernel_stats()
        new_stats = get_kernel_stats()
        
        assert new_stats.total_invocations == 0
        assert len(new_stats.kernel_counts) == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_attention_pipeline(self) -> None:
        """Test complete attention pipeline."""
        dispatcher = HopperKernelDispatcher()
        
        batch, heads, seq_len, head_dim = 2, 4, 64, 32
        
        q = torch.randn(batch, heads, seq_len, head_dim)
        k = torch.randn(batch, heads, seq_len, head_dim)
        v = torch.randn(batch, heads, seq_len, head_dim)
        
        # Run attention
        attn_output = dispatcher.attention(q, k, v)
        
        # Apply RMS norm
        weight = torch.ones(head_dim)
        output = dispatcher.rms_norm(
            attn_output.view(-1, head_dim),
            weight,
        )
        
        assert output.shape == (batch * heads * seq_len, head_dim)

    def test_ffn_pipeline(self) -> None:
        """Test feed-forward network pipeline."""
        dispatcher = HopperKernelDispatcher()
        
        batch, hidden = 4, 256
        intermediate = 512
        
        x = torch.randn(batch, hidden)
        
        # Up projection
        w_up = torch.randn(hidden, intermediate)
        w_gate = torch.randn(hidden, intermediate)
        
        up = dispatcher.matmul(x, w_up)
        gate = dispatcher.matmul(x, w_gate)
        
        # SwiGLU activation (simulated)
        activated = torch.nn.functional.silu(gate) * up
        
        # Down projection
        w_down = torch.randn(intermediate, hidden)
        output = dispatcher.matmul(activated, w_down)
        
        assert output.shape == (batch, hidden)
