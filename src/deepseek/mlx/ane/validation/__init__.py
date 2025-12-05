"""
ANE Validation Module

Phase 7 implementation providing:
- Numerical validation (ANE vs PyTorch reference)
- Perplexity testing (WikiText-2, C4)
- Task accuracy evaluation (MMLU, HumanEval)
- Output distribution comparison (KL divergence)
- Performance benchmarks
"""

from .numerical import (
    NumericalValidator,
    NumericalValidationConfig,
    ValidationResult,
    compare_tensors,
    compare_models,
)
from .perplexity import (
    PerplexityEvaluator,
    PerplexityConfig,
    PerplexityResult,
)
from .distribution import (
    DistributionComparator,
    DistributionConfig,
    DistributionResult,
    kl_divergence,
    js_divergence,
)
from .benchmarks import (
    PerformanceBenchmark,
    BenchmarkConfig,
    BenchmarkResult,
    LatencyBenchmark,
    MemoryBenchmark,
    ThroughputBenchmark,
)
from .task_accuracy import (
    TaskEvaluator,
    MMLUEvaluator,
    HumanEvalEvaluator,
    TaskConfig,
    TaskResult,
)

__all__ = [
    # Numerical validation
    "NumericalValidator",
    "NumericalValidationConfig",
    "ValidationResult",
    "compare_tensors",
    "compare_models",
    # Perplexity
    "PerplexityEvaluator",
    "PerplexityConfig",
    "PerplexityResult",
    # Distribution
    "DistributionComparator",
    "DistributionConfig",
    "DistributionResult",
    "kl_divergence",
    "js_divergence",
    # Benchmarks
    "PerformanceBenchmark",
    "BenchmarkConfig",
    "BenchmarkResult",
    "LatencyBenchmark",
    "MemoryBenchmark",
    "ThroughputBenchmark",
    # Task Accuracy
    "TaskEvaluator",
    "MMLUEvaluator",
    "HumanEvalEvaluator",
    "TaskConfig",
    "TaskResult",
]
