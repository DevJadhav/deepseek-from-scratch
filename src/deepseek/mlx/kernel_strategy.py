"""
JIT vs AOT Kernel Strategy Analysis for MLX and PyTorch

This module provides comprehensive analysis and implementation of
Just-In-Time (JIT) vs Ahead-Of-Time (AOT) kernel compilation strategies.

Strategy Summary:
| Backend | Strategy | Implementation | Reason |
|---------|----------|----------------|--------|
| Metal   | AOT      | Candle Metal   | Metal shaders are pre-compiled |
| CUDA    | AOT      | Candle CUDA    | NVRTC compilation is expensive |
| Triton  | JIT      | Python Triton  | Specializes to input shapes |
| MLX     | JIT      | MLX compile    | Dynamic graph optimization |

Trade-offs:

AOT (Ahead-Of-Time):
- Pros: No runtime compilation overhead, deterministic performance
- Cons: Generic kernels, no shape specialization, larger binary

JIT (Just-In-Time):
- Pros: Shape specialization, optimal register usage, fusion opportunities
- Cons: First-run compilation, cache management, non-determinism
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

LOGGER = logging.getLogger(__name__)

# Try to import MLX
MLX_AVAILABLE = False
try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except ImportError:
    mx = None


# =============================================================================
# Compilation Strategy Types
# =============================================================================


class CompilationStrategy(Enum):
    """Kernel compilation strategy."""

    AOT = "aot"  # Ahead-of-time compilation
    JIT = "jit"  # Just-in-time compilation
    HYBRID = "hybrid"  # AOT for common shapes, JIT for uncommon


class Backend(Enum):
    """Compute backend type."""

    METAL = "metal"
    CUDA = "cuda"
    TRITON = "triton"
    MLX = "mlx"
    CPU = "cpu"

    def default_strategy(self) -> CompilationStrategy:
        """Get default compilation strategy for this backend."""
        if self in (Backend.METAL, Backend.CUDA, Backend.CPU):
            return CompilationStrategy.AOT
        else:
            return CompilationStrategy.JIT

    def supports_jit(self) -> bool:
        """Check if backend supports JIT compilation."""
        return self in (Backend.TRITON, Backend.MLX)

    def supports_shape_specialization(self) -> bool:
        """Check if backend can specialize on input shapes."""
        return self in (Backend.TRITON, Backend.MLX)


# =============================================================================
# Kernel Key and Caching
# =============================================================================


@dataclass(frozen=True)
class KernelKey:
    """Unique key for a compiled kernel."""

    name: str
    shapes: tuple[tuple[int, ...], ...]
    dtype: str
    options: str = ""

    @classmethod
    def create(
        cls,
        name: str,
        shapes: list[tuple[int, ...] | list[int]],
        dtype: str,
        options: str = "",
    ) -> "KernelKey":
        """Create a kernel key from shapes."""
        normalized_shapes = tuple(tuple(s) for s in shapes)
        return cls(name=name, shapes=normalized_shapes, dtype=dtype, options=options)


@dataclass
class CompiledKernel:
    """Information about a compiled kernel."""

    key: KernelKey
    compile_time_ms: float
    is_jit: bool
    invocation_count: int = 0
    total_exec_time_ms: float = 0.0
    last_exec_time_ms: float = 0.0

    def record_invocation(self, exec_time_ms: float) -> None:
        """Record an execution of this kernel."""
        self.invocation_count += 1
        self.total_exec_time_ms += exec_time_ms
        self.last_exec_time_ms = exec_time_ms

    def avg_exec_time_ms(self) -> float:
        """Get average execution time."""
        if self.invocation_count == 0:
            return 0.0
        return self.total_exec_time_ms / self.invocation_count

    def is_amortized(self, threshold: int = 10) -> bool:
        """Check if JIT compilation cost has been amortized."""
        return self.invocation_count >= threshold

    def total_time_ms(self) -> float:
        """Get total time including compilation."""
        return self.compile_time_ms + self.total_exec_time_ms


@dataclass
class CacheStats:
    """Statistics for kernel cache."""

    size: int
    max_size: int
    hits: int
    misses: int

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class KernelCache:
    """Cache for compiled kernels."""

    def __init__(self, max_size: int = 1024):
        """Initialize cache."""
        self.max_size = max_size
        self._cache: dict[KernelKey, CompiledKernel] = {}
        self._hits = 0
        self._misses = 0

    def get(self, key: KernelKey) -> CompiledKernel | None:
        """Get a cached kernel."""
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def insert(self, kernel: CompiledKernel) -> None:
        """Insert a kernel into the cache."""
        if len(self._cache) >= self.max_size:
            # Evict least-used kernel
            min_key = min(self._cache.keys(), key=lambda k: self._cache[k].invocation_count)
            del self._cache[min_key]
        self._cache[kernel.key] = kernel

    def update_stats(self, key: KernelKey, exec_time_ms: float) -> None:
        """Update execution stats for a kernel."""
        if key in self._cache:
            self._cache[key].record_invocation(exec_time_ms)

    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return CacheStats(
            size=len(self._cache),
            max_size=self.max_size,
            hits=self._hits,
            misses=self._misses,
        )

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0


# =============================================================================
# Backend-Specific Configurations
# =============================================================================


@dataclass
class MetalAOTConfig:
    """Metal AOT compilation configuration."""

    enable_caching: bool = True
    precompile_variants: bool = True
    precompile_tile_sizes: list[tuple[int, int, int]] = field(
        default_factory=lambda: [(16, 16, 16), (32, 32, 32), (64, 64, 32)]
    )


@dataclass
class CudaAOTConfig:
    """CUDA AOT compilation configuration."""

    enable_caching: bool = True
    use_cubin_cache: bool = True
    target_sm_versions: list[tuple[int, int]] = field(
        default_factory=lambda: [(7, 0), (8, 0), (9, 0)]
    )
    use_tma: bool = True
    use_warpgroup: bool = True


@dataclass
class TritonJITConfig:
    """Triton JIT compilation configuration."""

    enable_caching: bool = True
    cache_dir: str | None = None
    enable_autotune: bool = True
    autotune_warmup: int = 25
    autotune_reps: int = 100


@dataclass
class MlxJITConfig:
    """MLX JIT compilation configuration."""

    enable_caching: bool = True
    enable_fusion: bool = True
    enable_constant_folding: bool = True
    compile_mode: str = "lazy"  # "lazy" or "eager"


@dataclass
class KernelStrategyConfig:
    """Unified kernel strategy configuration."""

    metal: MetalAOTConfig = field(default_factory=MetalAOTConfig)
    cuda: CudaAOTConfig = field(default_factory=CudaAOTConfig)
    triton: TritonJITConfig = field(default_factory=TritonJITConfig)
    mlx: MlxJITConfig = field(default_factory=MlxJITConfig)
    enable_hybrid: bool = True
    jit_amortization_threshold: int = 10


# =============================================================================
# Strategy Analysis
# =============================================================================


@dataclass
class StrategyAnalysis:
    """Analysis result for kernel strategy selection."""

    recommended: CompilationStrategy
    reasoning: list[str]
    expected_compile_time_ms: float | None = None
    expected_jit_speedup: float | None = None
    jit_break_even_invocations: int | None = None

    @classmethod
    def analyze(cls, backend: Backend, key: KernelKey) -> "StrategyAnalysis":
        """Analyze and recommend a strategy for a kernel."""
        reasoning = []
        recommended = backend.default_strategy()
        expected_jit_speedup = None
        jit_break_even = None
        expected_compile_time_ms = None

        if backend == Backend.METAL:
            reasoning.extend([
                "Metal uses AOT compilation via MTLComputePipelineState",
                "Metal shaders are pre-compiled into metallib",
                "Pipeline state caching eliminates recompilation",
            ])
            expected_compile_time_ms = 0.1  # Already compiled

        elif backend == Backend.CUDA:
            reasoning.extend([
                "CUDA uses AOT via NVCC or cubin caching",
                "NVRTC JIT compilation is expensive (100s of ms)",
            ])
            expected_compile_time_ms = 1.0  # Cached cubin

            # Check if JIT would help
            max_dim = max(max(s) for s in key.shapes) if key.shapes else 0
            if max_dim > 4096:
                reasoning.append("Large tensors benefit from shape-specialized tiling")
                expected_jit_speedup = 1.1  # 10% speedup
                jit_break_even = 100

        elif backend == Backend.TRITON:
            reasoning.extend([
                "Triton is designed for JIT compilation",
                "Shape specialization can yield 2-3x speedup",
                "Auto-tuning selects optimal tile sizes",
            ])
            recommended = CompilationStrategy.JIT
            expected_jit_speedup = 2.0
            jit_break_even = 50
            expected_compile_time_ms = 100.0  # First-run compilation

        elif backend == Backend.MLX:
            reasoning.extend([
                "MLX uses lazy JIT compilation",
                "Graph fusion optimizes memory bandwidth",
                "Compilation is fast (~10ms)",
            ])
            recommended = CompilationStrategy.JIT
            expected_jit_speedup = 1.5
            jit_break_even = 10
            expected_compile_time_ms = 10.0

        elif backend == Backend.CPU:
            reasoning.append("CPU uses pre-compiled BLAS/LAPACK kernels")
            expected_compile_time_ms = 0.0

        return cls(
            recommended=recommended,
            reasoning=reasoning,
            expected_compile_time_ms=expected_compile_time_ms,
            expected_jit_speedup=expected_jit_speedup,
            jit_break_even_invocations=jit_break_even,
        )


# =============================================================================
# Kernel Strategy Manager
# =============================================================================


class KernelStrategyManager:
    """Manages kernel compilation strategy selection and caching."""

    def __init__(self, config: KernelStrategyConfig | None = None):
        """Initialize manager."""
        self.config = config or KernelStrategyConfig()
        self.cache = KernelCache()
        self._backend_caches: dict[Backend, KernelCache] = {
            backend: KernelCache(256) for backend in Backend
        }

    def decide_strategy(self, backend: Backend, key: KernelKey) -> CompilationStrategy:
        """Decide compilation strategy for a kernel."""
        if not self.config.enable_hybrid:
            return backend.default_strategy()

        # Check cache
        cached = self.cache.get(key)
        if cached and cached.is_amortized(self.config.jit_amortization_threshold):
            return CompilationStrategy.JIT

        if backend.supports_jit():
            if self._is_common_shape(key.shapes):
                return CompilationStrategy.AOT
            return CompilationStrategy.JIT

        return CompilationStrategy.AOT

    def _is_common_shape(self, shapes: tuple[tuple[int, ...], ...]) -> bool:
        """Check if shapes are common (suitable for AOT)."""
        for shape in shapes:
            for dim in shape:
                if dim > 8192 or not _is_nice_number(dim):
                    return False
        return True

    def get_kernel(self, backend: Backend, key: KernelKey) -> CompiledKernel:
        """Get or compile a kernel."""
        cached = self.cache.get(key)
        if cached:
            return cached

        # Simulate compilation
        strategy = self.decide_strategy(backend, key)
        start = time.perf_counter()

        # In real implementation, this would call backend-specific compilers
        compile_time_ms = (time.perf_counter() - start) * 1000

        kernel = CompiledKernel(
            key=key,
            compile_time_ms=compile_time_ms,
            is_jit=(strategy == CompilationStrategy.JIT),
        )

        self.cache.insert(kernel)
        return kernel

    def record_execution(self, key: KernelKey, exec_time_ms: float) -> None:
        """Record kernel execution time."""
        self.cache.update_stats(key, exec_time_ms)

    def stats(self) -> dict[str, CacheStats]:
        """Get all cache statistics."""
        stats = {"global": self.cache.stats()}
        for backend, cache in self._backend_caches.items():
            stats[backend.value] = cache.stats()
        return stats


def _is_nice_number(n: int) -> bool:
    """Check if number is 'nice' for kernel dimensions."""
    if n == 0:
        return False
    # Power of 2
    if (n & (n - 1)) == 0:
        return True
    # Multiple of 32 or 64
    if n % 64 == 0 or n % 32 == 0:
        return True
    # Common dimensions
    if n in (128, 256, 512, 1024, 2048, 4096):
        return True
    return False


# =============================================================================
# MLX JIT Compilation Utilities
# =============================================================================


if MLX_AVAILABLE:

    def mlx_compile_function(
        fn: Any,
        inputs: Any = None,
        shapeless: bool = False,
    ) -> Any:
        """
        Compile an MLX function for optimized execution.

        MLX uses lazy JIT compilation:
        - First call traces the computation graph
        - Graph is optimized (fusion, constant folding)
        - Compiled graph is cached and reused

        Args:
            fn: Function to compile
            inputs: Optional sample inputs for tracing
            shapeless: If True, compile for any input shape

        Returns:
            Compiled function
        """
        # MLX's compile decorator
        compiled = mx.compile(fn, inputs=inputs, shapeless=shapeless)
        return compiled

    def mlx_eval_compiled(
        *arrays: Any,
    ) -> None:
        """
        Force evaluation of MLX arrays.

        This triggers JIT compilation and execution of the
        computation graph for the given arrays.

        Args:
            *arrays: Arrays to evaluate
        """
        mx.eval(*arrays)

    class MLXKernelStrategy:
        """MLX-specific kernel strategy management."""

        def __init__(self, config: MlxJITConfig | None = None):
            """Initialize MLX kernel strategy."""
            self.config = config or MlxJITConfig()
            self._compiled_functions: dict[str, Any] = {}
            self._compile_times: dict[str, float] = {}

        def compile(self, name: str, fn: Any, **kwargs: Any) -> Any:
            """Compile and cache a function."""
            if name in self._compiled_functions:
                return self._compiled_functions[name]

            start = time.perf_counter()
            compiled = mx.compile(fn, **kwargs)
            compile_time = (time.perf_counter() - start) * 1000

            self._compiled_functions[name] = compiled
            self._compile_times[name] = compile_time

            LOGGER.debug(f"Compiled MLX function '{name}' in {compile_time:.2f}ms")
            return compiled

        def get_compile_times(self) -> dict[str, float]:
            """Get compilation times for all functions."""
            return self._compile_times.copy()


# Global strategy manager
_strategy_manager: KernelStrategyManager | None = None


def get_strategy_manager() -> KernelStrategyManager:
    """Get global kernel strategy manager."""
    global _strategy_manager
    if _strategy_manager is None:
        _strategy_manager = KernelStrategyManager()
    return _strategy_manager


__all__ = [
    # Types
    "CompilationStrategy",
    "Backend",
    "KernelKey",
    "CompiledKernel",
    "CacheStats",
    "KernelCache",
    # Configs
    "MetalAOTConfig",
    "CudaAOTConfig",
    "TritonJITConfig",
    "MlxJITConfig",
    "KernelStrategyConfig",
    # Analysis
    "StrategyAnalysis",
    "KernelStrategyManager",
    "get_strategy_manager",
]

if MLX_AVAILABLE:
    __all__.extend([
        "mlx_compile_function",
        "mlx_eval_compiled",
        "MLXKernelStrategy",
    ])
