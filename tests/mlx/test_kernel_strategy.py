"""
Tests for JIT vs AOT Kernel Strategy Analysis

These tests verify the kernel compilation strategy implementations for:
- Strategy selection (AOT, JIT, Hybrid)
- Kernel caching
- Backend-specific configurations
- Strategy analysis and recommendations
"""

import pytest
import time

from deepseek.mlx.kernel_strategy import (
    Backend,
    CacheStats,
    CompiledKernel,
    CompilationStrategy,
    CudaAOTConfig,
    KernelCache,
    KernelKey,
    KernelStrategyConfig,
    KernelStrategyManager,
    MetalAOTConfig,
    MlxJITConfig,
    StrategyAnalysis,
    TritonJITConfig,
    get_strategy_manager,
)

# Check MLX availability
try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


# =============================================================================
# Compilation Strategy Tests
# =============================================================================


class TestCompilationStrategy:
    """Tests for CompilationStrategy enum."""

    def test_strategy_values(self) -> None:
        """Test strategy enum values."""
        assert CompilationStrategy.AOT.value == "aot"
        assert CompilationStrategy.JIT.value == "jit"
        assert CompilationStrategy.HYBRID.value == "hybrid"


# =============================================================================
# Backend Tests
# =============================================================================


class TestBackend:
    """Tests for Backend enum."""

    def test_backend_values(self) -> None:
        """Test backend enum values."""
        assert Backend.METAL.value == "metal"
        assert Backend.CUDA.value == "cuda"
        assert Backend.TRITON.value == "triton"
        assert Backend.MLX.value == "mlx"
        assert Backend.CPU.value == "cpu"

    def test_default_strategy(self) -> None:
        """Test default strategy selection."""
        assert Backend.METAL.default_strategy() == CompilationStrategy.AOT
        assert Backend.CUDA.default_strategy() == CompilationStrategy.AOT
        assert Backend.TRITON.default_strategy() == CompilationStrategy.JIT
        assert Backend.MLX.default_strategy() == CompilationStrategy.JIT
        assert Backend.CPU.default_strategy() == CompilationStrategy.AOT

    def test_supports_jit(self) -> None:
        """Test JIT support detection."""
        assert not Backend.METAL.supports_jit()
        assert not Backend.CUDA.supports_jit()
        assert Backend.TRITON.supports_jit()
        assert Backend.MLX.supports_jit()
        assert not Backend.CPU.supports_jit()

    def test_supports_shape_specialization(self) -> None:
        """Test shape specialization support."""
        assert not Backend.METAL.supports_shape_specialization()
        assert not Backend.CUDA.supports_shape_specialization()
        assert Backend.TRITON.supports_shape_specialization()
        assert Backend.MLX.supports_shape_specialization()


# =============================================================================
# Kernel Key Tests
# =============================================================================


class TestKernelKey:
    """Tests for KernelKey."""

    def test_create(self) -> None:
        """Test creating kernel key."""
        key = KernelKey.create(
            name="matmul",
            shapes=[(1024, 1024), (1024, 1024)],
            dtype="f32",
        )
        
        assert key.name == "matmul"
        assert key.shapes == ((1024, 1024), (1024, 1024))
        assert key.dtype == "f32"
        assert key.options == ""

    def test_create_with_options(self) -> None:
        """Test creating kernel key with options."""
        key = KernelKey.create(
            name="attention",
            shapes=[(2, 8, 512, 64)],
            dtype="f16",
            options="causal=True",
        )
        
        assert key.options == "causal=True"

    def test_hashable(self) -> None:
        """Test that kernel keys are hashable."""
        key1 = KernelKey.create("test", [(32, 32)], "f32")
        key2 = KernelKey.create("test", [(32, 32)], "f32")
        
        assert hash(key1) == hash(key2)
        assert key1 == key2

    def test_different_keys(self) -> None:
        """Test that different configurations produce different keys."""
        key1 = KernelKey.create("matmul", [(32, 32)], "f32")
        key2 = KernelKey.create("matmul", [(64, 64)], "f32")
        key3 = KernelKey.create("matmul", [(32, 32)], "f16")
        
        assert key1 != key2
        assert key1 != key3


# =============================================================================
# Compiled Kernel Tests
# =============================================================================


class TestCompiledKernel:
    """Tests for CompiledKernel."""

    def test_creation(self) -> None:
        """Test creating compiled kernel."""
        key = KernelKey.create("test", [(32, 32)], "f32")
        kernel = CompiledKernel(
            key=key,
            compile_time_ms=10.0,
            is_jit=True,
        )
        
        assert kernel.key == key
        assert kernel.compile_time_ms == 10.0
        assert kernel.is_jit
        assert kernel.invocation_count == 0
        assert kernel.total_exec_time_ms == 0.0

    def test_record_invocation(self) -> None:
        """Test recording invocations."""
        key = KernelKey.create("test", [(32, 32)], "f32")
        kernel = CompiledKernel(key=key, compile_time_ms=10.0, is_jit=True)
        
        kernel.record_invocation(1.5)
        kernel.record_invocation(2.0)
        kernel.record_invocation(1.0)
        
        assert kernel.invocation_count == 3
        assert kernel.total_exec_time_ms == 4.5
        assert kernel.last_exec_time_ms == 1.0

    def test_avg_exec_time(self) -> None:
        """Test average execution time calculation."""
        key = KernelKey.create("test", [(32, 32)], "f32")
        kernel = CompiledKernel(key=key, compile_time_ms=10.0, is_jit=True)
        
        # No invocations
        assert kernel.avg_exec_time_ms() == 0.0
        
        # With invocations
        kernel.record_invocation(1.0)
        kernel.record_invocation(3.0)
        
        assert kernel.avg_exec_time_ms() == 2.0

    def test_is_amortized(self) -> None:
        """Test amortization check."""
        key = KernelKey.create("test", [(32, 32)], "f32")
        kernel = CompiledKernel(key=key, compile_time_ms=100.0, is_jit=True)
        
        # Not amortized yet
        assert not kernel.is_amortized(threshold=10)
        
        # Add invocations
        for _ in range(10):
            kernel.record_invocation(1.0)
        
        # Now amortized
        assert kernel.is_amortized(threshold=10)

    def test_total_time(self) -> None:
        """Test total time calculation."""
        key = KernelKey.create("test", [(32, 32)], "f32")
        kernel = CompiledKernel(key=key, compile_time_ms=100.0, is_jit=True)
        
        kernel.record_invocation(10.0)
        kernel.record_invocation(20.0)
        
        assert kernel.total_time_ms() == 130.0  # 100 + 10 + 20


# =============================================================================
# Kernel Cache Tests
# =============================================================================


class TestKernelCache:
    """Tests for KernelCache."""

    def test_empty_cache(self) -> None:
        """Test empty cache behavior."""
        cache = KernelCache(max_size=10)
        key = KernelKey.create("test", [(32, 32)], "f32")
        
        assert cache.get(key) is None

    def test_insert_and_get(self) -> None:
        """Test inserting and retrieving kernels."""
        cache = KernelCache(max_size=10)
        key = KernelKey.create("test", [(32, 32)], "f32")
        kernel = CompiledKernel(key=key, compile_time_ms=10.0, is_jit=True)
        
        cache.insert(kernel)
        
        result = cache.get(key)
        assert result is not None
        assert result.key == key

    def test_cache_stats(self) -> None:
        """Test cache statistics."""
        cache = KernelCache(max_size=10)
        key = KernelKey.create("test", [(32, 32)], "f32")
        kernel = CompiledKernel(key=key, compile_time_ms=10.0, is_jit=True)
        
        # Miss
        cache.get(key)
        
        # Insert
        cache.insert(kernel)
        
        # Hit
        cache.get(key)
        
        stats = cache.stats()
        assert stats.size == 1
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.hit_rate == 0.5

    def test_cache_eviction(self) -> None:
        """Test cache eviction when full."""
        cache = KernelCache(max_size=2)
        
        # Insert 3 kernels
        for i in range(3):
            key = KernelKey.create(f"test_{i}", [(32, 32)], "f32")
            kernel = CompiledKernel(key=key, compile_time_ms=10.0, is_jit=True)
            cache.insert(kernel)
        
        stats = cache.stats()
        assert stats.size == 2  # Only 2 should remain

    def test_update_stats(self) -> None:
        """Test updating execution stats."""
        cache = KernelCache(max_size=10)
        key = KernelKey.create("test", [(32, 32)], "f32")
        kernel = CompiledKernel(key=key, compile_time_ms=10.0, is_jit=True)
        
        cache.insert(kernel)
        cache.update_stats(key, exec_time_ms=5.0)
        
        result = cache.get(key)
        assert result.invocation_count == 1
        assert result.total_exec_time_ms == 5.0

    def test_clear(self) -> None:
        """Test clearing cache."""
        cache = KernelCache(max_size=10)
        key = KernelKey.create("test", [(32, 32)], "f32")
        kernel = CompiledKernel(key=key, compile_time_ms=10.0, is_jit=True)
        
        cache.insert(kernel)
        cache.clear()
        
        stats = cache.stats()
        assert stats.size == 0
        assert stats.hits == 0
        assert stats.misses == 0


# =============================================================================
# Configuration Tests
# =============================================================================


class TestMetalAOTConfig:
    """Tests for Metal AOT configuration."""

    def test_default(self) -> None:
        """Test default configuration."""
        config = MetalAOTConfig()
        
        assert config.enable_caching
        assert config.precompile_variants
        assert len(config.precompile_tile_sizes) > 0


class TestCudaAOTConfig:
    """Tests for CUDA AOT configuration."""

    def test_default(self) -> None:
        """Test default configuration."""
        config = CudaAOTConfig()
        
        assert config.enable_caching
        assert config.use_cubin_cache
        assert config.use_tma
        assert config.use_warpgroup
        assert len(config.target_sm_versions) > 0


class TestTritonJITConfig:
    """Tests for Triton JIT configuration."""

    def test_default(self) -> None:
        """Test default configuration."""
        config = TritonJITConfig()
        
        assert config.enable_caching
        assert config.enable_autotune
        assert config.autotune_warmup > 0
        assert config.autotune_reps > 0


class TestMlxJITConfig:
    """Tests for MLX JIT configuration."""

    def test_default(self) -> None:
        """Test default configuration."""
        config = MlxJITConfig()
        
        assert config.enable_caching
        assert config.enable_fusion
        assert config.enable_constant_folding
        assert config.compile_mode == "lazy"


class TestKernelStrategyConfig:
    """Tests for unified strategy configuration."""

    def test_default(self) -> None:
        """Test default configuration."""
        config = KernelStrategyConfig()
        
        assert isinstance(config.metal, MetalAOTConfig)
        assert isinstance(config.cuda, CudaAOTConfig)
        assert isinstance(config.triton, TritonJITConfig)
        assert isinstance(config.mlx, MlxJITConfig)
        assert config.enable_hybrid
        assert config.jit_amortization_threshold > 0


# =============================================================================
# Strategy Analysis Tests
# =============================================================================


class TestStrategyAnalysis:
    """Tests for strategy analysis."""

    def test_metal_analysis(self) -> None:
        """Test Metal strategy analysis."""
        key = KernelKey.create("matmul", [(1024, 1024)], "f32")
        analysis = StrategyAnalysis.analyze(Backend.METAL, key)
        
        assert analysis.recommended == CompilationStrategy.AOT
        assert len(analysis.reasoning) > 0
        assert analysis.expected_compile_time_ms is not None

    def test_cuda_analysis(self) -> None:
        """Test CUDA strategy analysis."""
        key = KernelKey.create("matmul", [(1024, 1024)], "f32")
        analysis = StrategyAnalysis.analyze(Backend.CUDA, key)
        
        assert analysis.recommended == CompilationStrategy.AOT
        assert len(analysis.reasoning) > 0

    def test_triton_analysis(self) -> None:
        """Test Triton strategy analysis."""
        key = KernelKey.create("attention", [(2048, 64)], "f32")
        analysis = StrategyAnalysis.analyze(Backend.TRITON, key)
        
        assert analysis.recommended == CompilationStrategy.JIT
        assert analysis.expected_jit_speedup is not None
        assert analysis.jit_break_even_invocations is not None

    def test_mlx_analysis(self) -> None:
        """Test MLX strategy analysis."""
        key = KernelKey.create("softmax", [(4096,)], "f32")
        analysis = StrategyAnalysis.analyze(Backend.MLX, key)
        
        assert analysis.recommended == CompilationStrategy.JIT
        assert analysis.expected_jit_speedup is not None

    def test_cpu_analysis(self) -> None:
        """Test CPU strategy analysis."""
        key = KernelKey.create("matmul", [(256, 256)], "f32")
        analysis = StrategyAnalysis.analyze(Backend.CPU, key)
        
        assert analysis.recommended == CompilationStrategy.AOT


# =============================================================================
# Kernel Strategy Manager Tests
# =============================================================================


class TestKernelStrategyManager:
    """Tests for KernelStrategyManager."""

    def test_creation(self) -> None:
        """Test manager creation."""
        manager = KernelStrategyManager()
        
        assert manager.config is not None
        assert manager.cache is not None

    def test_decide_strategy_aot(self) -> None:
        """Test strategy decision for AOT backend."""
        manager = KernelStrategyManager()
        key = KernelKey.create("test", [(32, 32)], "f32")
        
        strategy = manager.decide_strategy(Backend.METAL, key)
        
        assert strategy == CompilationStrategy.AOT

    def test_decide_strategy_jit(self) -> None:
        """Test strategy decision for JIT backend."""
        manager = KernelStrategyManager()
        
        # Uncommon shape should trigger JIT for JIT-capable backends
        key = KernelKey.create("test", [(12345, 6789)], "f32")
        
        strategy = manager.decide_strategy(Backend.TRITON, key)
        
        assert strategy == CompilationStrategy.JIT

    def test_get_kernel(self) -> None:
        """Test getting/compiling kernel."""
        manager = KernelStrategyManager()
        key = KernelKey.create("test", [(32, 32)], "f32")
        
        kernel = manager.get_kernel(Backend.CUDA, key)
        
        assert kernel.key == key

    def test_record_execution(self) -> None:
        """Test recording kernel execution."""
        manager = KernelStrategyManager()
        key = KernelKey.create("test", [(32, 32)], "f32")
        
        # First, get the kernel to register it
        manager.get_kernel(Backend.CUDA, key)
        
        # Record execution
        manager.record_execution(key, exec_time_ms=1.5)
        
        # Check stats
        stats = manager.stats()
        assert "global" in stats

    def test_stats(self) -> None:
        """Test getting statistics."""
        manager = KernelStrategyManager()
        
        stats = manager.stats()
        
        assert "global" in stats
        assert "metal" in stats
        assert "cuda" in stats


class TestGlobalStrategyManager:
    """Tests for global strategy manager."""

    def test_get_strategy_manager(self) -> None:
        """Test getting global manager."""
        manager = get_strategy_manager()
        
        assert manager is not None
        assert isinstance(manager, KernelStrategyManager)

    def test_singleton(self) -> None:
        """Test that manager is a singleton."""
        manager1 = get_strategy_manager()
        manager2 = get_strategy_manager()
        
        assert manager1 is manager2


# =============================================================================
# MLX-Specific Tests
# =============================================================================


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestMLXKernelStrategy:
    """Tests for MLX-specific kernel strategy."""

    def test_import_mlx_functions(self) -> None:
        """Test importing MLX-specific functions."""
        from deepseek.mlx.kernel_strategy import (
            MLXKernelStrategy,
            mlx_compile_function,
            mlx_eval_compiled,
        )
        
        assert mlx_compile_function is not None
        assert mlx_eval_compiled is not None
        assert MLXKernelStrategy is not None

    def test_mlx_kernel_strategy_creation(self) -> None:
        """Test MLX kernel strategy creation."""
        from deepseek.mlx.kernel_strategy import MLXKernelStrategy
        
        strategy = MLXKernelStrategy()
        assert strategy.config is not None

    def test_compile_function(self) -> None:
        """Test compiling a function."""
        from deepseek.mlx.kernel_strategy import MLXKernelStrategy
        
        def simple_fn(x):
            return x * 2
        
        strategy = MLXKernelStrategy()
        compiled = strategy.compile("simple", simple_fn)
        
        assert compiled is not None
        
        # Test execution
        x = mx.array([1.0, 2.0, 3.0])
        result = compiled(x)
        mx.eval(result)
        
        expected = mx.array([2.0, 4.0, 6.0])
        assert mx.allclose(result, expected)

    def test_compile_times(self) -> None:
        """Test compile time tracking."""
        from deepseek.mlx.kernel_strategy import MLXKernelStrategy
        
        def test_fn(x):
            return mx.softmax(x)
        
        strategy = MLXKernelStrategy()
        strategy.compile("test", test_fn)
        
        times = strategy.get_compile_times()
        assert "test" in times


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for kernel strategy system."""

    def test_full_workflow(self) -> None:
        """Test complete workflow."""
        manager = KernelStrategyManager()
        
        # Create kernel key
        key = KernelKey.create("attention", [(2, 8, 512, 64)], "f32")
        
        # Analyze strategy
        analysis = StrategyAnalysis.analyze(Backend.TRITON, key)
        
        # Get kernel
        kernel = manager.get_kernel(Backend.TRITON, key)
        
        # Record executions
        for _ in range(20):
            manager.record_execution(key, exec_time_ms=0.5)
        
        # Check amortization
        cached = manager.cache.get(key)
        assert cached is not None
        assert cached.is_amortized(threshold=10)

    def test_hybrid_strategy_decision(self) -> None:
        """Test hybrid strategy decision making."""
        config = KernelStrategyConfig(enable_hybrid=True)
        manager = KernelStrategyManager(config)
        
        # Common shape - should use AOT even for JIT backends
        common_key = KernelKey.create("test", [(1024, 1024)], "f32")
        
        # Uncommon shape - should use JIT for JIT backends
        uncommon_key = KernelKey.create("test", [(12345, 6789)], "f32")
        
        # For Triton (JIT backend)
        common_strategy = manager.decide_strategy(Backend.TRITON, common_key)
        uncommon_strategy = manager.decide_strategy(Backend.TRITON, uncommon_key)
        
        # Common shapes should prefer AOT
        assert common_strategy == CompilationStrategy.AOT
        # Uncommon shapes should use JIT
        assert uncommon_strategy == CompilationStrategy.JIT
