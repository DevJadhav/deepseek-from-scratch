"""
ANE Performance Benchmarks Module

Provides comprehensive performance benchmarking:
- Latency measurement (prefill, decode)
- Memory profiling (peak, average)
- Throughput measurement (tokens/second)
- ANE utilization estimation
"""

from __future__ import annotations

import gc
import json
import os
import statistics
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class BenchmarkType(Enum):
    """Types of benchmarks."""

    LATENCY = "latency"
    MEMORY = "memory"
    THROUGHPUT = "throughput"
    ANE_UTILIZATION = "ane_utilization"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarks."""

    # Warmup iterations
    num_warmup: int = 5

    # Benchmark iterations
    num_iterations: int = 100

    # Batch sizes to test
    batch_sizes: list[int] = field(default_factory=lambda: [1, 2, 4, 8])

    # Sequence lengths to test
    sequence_lengths: list[int] = field(default_factory=lambda: [128, 256, 512, 1024])

    # Device for benchmarking
    device: str = "cpu"

    # Enable memory tracking
    track_memory: bool = True

    # Save results to file
    save_results: bool = True
    results_dir: str = "benchmark_results"

    # Verbose output
    verbose: bool = False


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    # Benchmark type
    benchmark_type: str

    # Configuration used
    config_used: dict = field(default_factory=dict)

    # Latency metrics (ms)
    latency_mean_ms: float = 0.0
    latency_std_ms: float = 0.0
    latency_min_ms: float = 0.0
    latency_max_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0

    # Memory metrics (MB)
    memory_peak_mb: float = 0.0
    memory_allocated_mb: float = 0.0
    memory_reserved_mb: float = 0.0

    # Throughput metrics
    throughput_tokens_per_sec: float = 0.0
    throughput_samples_per_sec: float = 0.0

    # ANE metrics
    ane_utilization_percent: float = 0.0

    # Additional metrics
    extra_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "benchmark_type": self.benchmark_type,
            "config_used": self.config_used,
            "latency": {
                "mean_ms": self.latency_mean_ms,
                "std_ms": self.latency_std_ms,
                "min_ms": self.latency_min_ms,
                "max_ms": self.latency_max_ms,
                "p50_ms": self.latency_p50_ms,
                "p95_ms": self.latency_p95_ms,
                "p99_ms": self.latency_p99_ms,
            },
            "memory": {
                "peak_mb": self.memory_peak_mb,
                "allocated_mb": self.memory_allocated_mb,
                "reserved_mb": self.memory_reserved_mb,
            },
            "throughput": {
                "tokens_per_sec": self.throughput_tokens_per_sec,
                "samples_per_sec": self.throughput_samples_per_sec,
            },
            "ane_utilization_percent": self.ane_utilization_percent,
            "extra_metrics": self.extra_metrics,
        }

    def save(self, path: str):
        """Save results to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def percentile(data: list[float], p: float) -> float:
    """Compute percentile of data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


class LatencyBenchmark:
    """
    Latency benchmark for model inference.

    Measures:
    - Prefill latency (processing input sequence)
    - Decode latency (generating tokens)
    - End-to-end latency
    """

    def __init__(self, config: BenchmarkConfig | None = None):
        """Initialize benchmark with configuration."""
        self.config = config or BenchmarkConfig()

    def benchmark_forward(
        self,
        model: nn.Module,
        input_generator,
        batch_size: int = 1,
        seq_length: int = 128,
    ) -> BenchmarkResult:
        """
        Benchmark forward pass latency.

        Args:
            model: Model to benchmark
            input_generator: Function that generates inputs
            batch_size: Batch size
            seq_length: Sequence length

        Returns:
            BenchmarkResult with latency metrics
        """
        model.eval()
        device = torch.device(self.config.device)
        model = model.to(device)

        latencies = []

        # Warmup
        for _ in range(self.config.num_warmup):
            inputs = input_generator(batch_size, seq_length)
            with torch.no_grad():
                _ = model(**inputs)

        # Synchronize if using GPU/MPS
        if device.type in ("cuda", "mps"):
            torch.cuda.synchronize() if device.type == "cuda" else torch.mps.synchronize()

        # Benchmark
        for _ in range(self.config.num_iterations):
            inputs = input_generator(batch_size, seq_length)

            # Synchronize before timing
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()

            start = time.perf_counter()

            with torch.no_grad():
                _ = model(**inputs)

            # Synchronize after
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()

            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms

        return BenchmarkResult(
            benchmark_type="forward_latency",
            config_used={"batch_size": batch_size, "seq_length": seq_length},
            latency_mean_ms=statistics.mean(latencies),
            latency_std_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            latency_min_ms=min(latencies),
            latency_max_ms=max(latencies),
            latency_p50_ms=percentile(latencies, 50),
            latency_p95_ms=percentile(latencies, 95),
            latency_p99_ms=percentile(latencies, 99),
        )

    def benchmark_prefill(
        self,
        model: nn.Module,
        input_generator,
        seq_lengths: list[int] | None = None,
    ) -> list[BenchmarkResult]:
        """
        Benchmark prefill latency for various sequence lengths.

        Args:
            model: Model to benchmark
            input_generator: Function that generates inputs
            seq_lengths: Sequence lengths to test

        Returns:
            List of BenchmarkResult for each sequence length
        """
        seq_lengths = seq_lengths or self.config.sequence_lengths
        results = []

        for seq_len in seq_lengths:
            result = self.benchmark_forward(model, input_generator, batch_size=1, seq_length=seq_len)
            result.benchmark_type = "prefill_latency"
            results.append(result)

            if self.config.verbose:
                print(f"Prefill {seq_len} tokens: {result.latency_mean_ms:.2f}ms")

        return results

    def benchmark_decode(
        self,
        model: nn.Module,
        input_generator,
        num_tokens: int = 100,
    ) -> BenchmarkResult:
        """
        Benchmark decode latency (single token generation).

        Args:
            model: Model to benchmark
            input_generator: Function that generates single-token inputs
            num_tokens: Number of tokens to generate

        Returns:
            BenchmarkResult with per-token latency
        """
        model.eval()
        device = torch.device(self.config.device)
        model = model.to(device)

        latencies = []

        # Warmup
        for _ in range(self.config.num_warmup):
            inputs = input_generator(batch_size=1, seq_length=1)
            with torch.no_grad():
                _ = model(**inputs)

        # Benchmark
        for _ in range(num_tokens):
            inputs = input_generator(batch_size=1, seq_length=1)

            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()

            start = time.perf_counter()

            with torch.no_grad():
                _ = model(**inputs)

            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()

            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        return BenchmarkResult(
            benchmark_type="decode_latency",
            config_used={"num_tokens": num_tokens},
            latency_mean_ms=statistics.mean(latencies),
            latency_std_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            latency_min_ms=min(latencies),
            latency_max_ms=max(latencies),
            latency_p50_ms=percentile(latencies, 50),
            latency_p95_ms=percentile(latencies, 95),
            latency_p99_ms=percentile(latencies, 99),
        )


class MemoryBenchmark:
    """
    Memory benchmark for model inference.

    Measures:
    - Peak memory usage
    - Allocated memory
    - Memory growth over time
    """

    def __init__(self, config: BenchmarkConfig | None = None):
        """Initialize benchmark with configuration."""
        self.config = config or BenchmarkConfig()

    def benchmark_memory(
        self,
        model: nn.Module,
        input_generator,
        batch_size: int = 1,
        seq_length: int = 128,
    ) -> BenchmarkResult:
        """
        Benchmark memory usage.

        Args:
            model: Model to benchmark
            input_generator: Function that generates inputs
            batch_size: Batch size
            seq_length: Sequence length

        Returns:
            BenchmarkResult with memory metrics
        """
        model.eval()
        device = torch.device(self.config.device)
        model = model.to(device)

        # Clear memory
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        elif device.type == "mps":
            torch.mps.empty_cache()

        # Measure baseline
        if device.type == "cuda":
            baseline_allocated = torch.cuda.memory_allocated() / 1024 / 1024
        else:
            baseline_allocated = 0.0

        # Run inference
        inputs = input_generator(batch_size, seq_length)

        with torch.no_grad():
            _ = model(**inputs)

        # Measure memory
        if device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            allocated_memory = torch.cuda.memory_allocated() / 1024 / 1024
            reserved_memory = torch.cuda.memory_reserved() / 1024 / 1024
        elif device.type == "mps":
            # MPS doesn't have detailed memory stats
            allocated_memory = torch.mps.current_allocated_memory() / 1024 / 1024
            peak_memory = allocated_memory  # Approximation
            reserved_memory = allocated_memory
        else:
            # For CPU, estimate from model parameters
            param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
            allocated_memory = param_memory
            peak_memory = param_memory * 2  # Rough estimate including activations
            reserved_memory = param_memory

        return BenchmarkResult(
            benchmark_type="memory",
            config_used={"batch_size": batch_size, "seq_length": seq_length},
            memory_peak_mb=peak_memory,
            memory_allocated_mb=allocated_memory - baseline_allocated,
            memory_reserved_mb=reserved_memory,
        )

    def benchmark_model_size(self, model: nn.Module) -> BenchmarkResult:
        """
        Measure model size in memory.

        Args:
            model: Model to measure

        Returns:
            BenchmarkResult with model size
        """
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())

        total_mb = (param_bytes + buffer_bytes) / 1024 / 1024

        return BenchmarkResult(
            benchmark_type="model_size",
            memory_allocated_mb=total_mb,
            extra_metrics={
                "num_parameters": sum(p.numel() for p in model.parameters()),
                "num_buffers": sum(b.numel() for b in model.buffers()),
                "param_bytes": param_bytes,
                "buffer_bytes": buffer_bytes,
            },
        )


class ThroughputBenchmark:
    """
    Throughput benchmark for model inference.

    Measures:
    - Tokens per second
    - Samples per second
    - Batch efficiency
    """

    def __init__(self, config: BenchmarkConfig | None = None):
        """Initialize benchmark with configuration."""
        self.config = config or BenchmarkConfig()

    def benchmark_throughput(
        self,
        model: nn.Module,
        input_generator,
        batch_size: int = 1,
        seq_length: int = 128,
        duration_sec: float = 10.0,
    ) -> BenchmarkResult:
        """
        Benchmark throughput.

        Args:
            model: Model to benchmark
            input_generator: Function that generates inputs
            batch_size: Batch size
            seq_length: Sequence length
            duration_sec: Duration to run benchmark

        Returns:
            BenchmarkResult with throughput metrics
        """
        model.eval()
        device = torch.device(self.config.device)
        model = model.to(device)

        # Warmup
        for _ in range(self.config.num_warmup):
            inputs = input_generator(batch_size, seq_length)
            with torch.no_grad():
                _ = model(**inputs)

        # Benchmark
        num_samples = 0
        num_tokens = 0

        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

        start_time = time.perf_counter()

        while time.perf_counter() - start_time < duration_sec:
            inputs = input_generator(batch_size, seq_length)

            with torch.no_grad():
                _ = model(**inputs)

            num_samples += batch_size
            num_tokens += batch_size * seq_length

        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

        elapsed_time = time.perf_counter() - start_time

        tokens_per_sec = num_tokens / elapsed_time
        samples_per_sec = num_samples / elapsed_time

        return BenchmarkResult(
            benchmark_type="throughput",
            config_used={
                "batch_size": batch_size,
                "seq_length": seq_length,
                "duration_sec": duration_sec,
            },
            throughput_tokens_per_sec=tokens_per_sec,
            throughput_samples_per_sec=samples_per_sec,
            extra_metrics={
                "total_tokens": num_tokens,
                "total_samples": num_samples,
                "elapsed_time_sec": elapsed_time,
            },
        )

    def benchmark_batch_scaling(
        self,
        model: nn.Module,
        input_generator,
        seq_length: int = 128,
        batch_sizes: list[int] | None = None,
    ) -> list[BenchmarkResult]:
        """
        Benchmark throughput scaling with batch size.

        Args:
            model: Model to benchmark
            input_generator: Function that generates inputs
            seq_length: Sequence length
            batch_sizes: Batch sizes to test

        Returns:
            List of BenchmarkResult for each batch size
        """
        batch_sizes = batch_sizes or self.config.batch_sizes
        results = []

        for batch_size in batch_sizes:
            result = self.benchmark_throughput(
                model, input_generator,
                batch_size=batch_size,
                seq_length=seq_length,
                duration_sec=5.0,
            )
            results.append(result)

            if self.config.verbose:
                print(
                    f"Batch {batch_size}: {result.throughput_tokens_per_sec:.0f} tok/s, "
                    f"{result.throughput_samples_per_sec:.1f} samples/s"
                )

        return results


class PerformanceBenchmark:
    """
    Comprehensive performance benchmark suite.

    Example:
        benchmark = PerformanceBenchmark(config)

        # Run all benchmarks
        results = benchmark.run_all(model, input_generator)

        # Run specific benchmarks
        latency_result = benchmark.benchmark_latency(model, input_generator)
        memory_result = benchmark.benchmark_memory(model, input_generator)
        throughput_result = benchmark.benchmark_throughput(model, input_generator)

        # Save results
        benchmark.save_results("benchmark_results.json")
    """

    def __init__(self, config: BenchmarkConfig | None = None):
        """Initialize benchmark suite."""
        self.config = config or BenchmarkConfig()
        self.latency = LatencyBenchmark(config)
        self.memory = MemoryBenchmark(config)
        self.throughput = ThroughputBenchmark(config)
        self.results: list[BenchmarkResult] = []

        if self.config.save_results:
            Path(self.config.results_dir).mkdir(parents=True, exist_ok=True)

    def benchmark_latency(
        self,
        model: nn.Module,
        input_generator,
        **kwargs,
    ) -> BenchmarkResult:
        """Run latency benchmark."""
        result = self.latency.benchmark_forward(model, input_generator, **kwargs)
        self.results.append(result)
        return result

    def benchmark_memory(
        self,
        model: nn.Module,
        input_generator,
        **kwargs,
    ) -> BenchmarkResult:
        """Run memory benchmark."""
        result = self.memory.benchmark_memory(model, input_generator, **kwargs)
        self.results.append(result)
        return result

    def benchmark_throughput(
        self,
        model: nn.Module,
        input_generator,
        **kwargs,
    ) -> BenchmarkResult:
        """Run throughput benchmark."""
        result = self.throughput.benchmark_throughput(model, input_generator, **kwargs)
        self.results.append(result)
        return result

    def run_all(
        self,
        model: nn.Module,
        input_generator,
        batch_size: int = 1,
        seq_length: int = 128,
    ) -> dict[str, BenchmarkResult]:
        """
        Run all benchmarks.

        Args:
            model: Model to benchmark
            input_generator: Function that generates inputs
            batch_size: Batch size
            seq_length: Sequence length

        Returns:
            Dictionary of benchmark results
        """
        results = {}

        if self.config.verbose:
            print("Running latency benchmark...")
        results["latency"] = self.benchmark_latency(
            model, input_generator, batch_size=batch_size, seq_length=seq_length
        )

        if self.config.track_memory:
            if self.config.verbose:
                print("Running memory benchmark...")
            results["memory"] = self.benchmark_memory(
                model, input_generator, batch_size=batch_size, seq_length=seq_length
            )

        if self.config.verbose:
            print("Running throughput benchmark...")
        results["throughput"] = self.benchmark_throughput(
            model, input_generator, batch_size=batch_size, seq_length=seq_length
        )

        return results

    def compare_models(
        self,
        models: dict[str, nn.Module],
        input_generator,
        batch_size: int = 1,
        seq_length: int = 128,
    ) -> dict[str, dict[str, BenchmarkResult]]:
        """
        Compare multiple models.

        Args:
            models: Dictionary of model name to model
            input_generator: Function that generates inputs
            batch_size: Batch size
            seq_length: Sequence length

        Returns:
            Nested dictionary of results by model and benchmark type
        """
        all_results = {}

        for name, model in models.items():
            if self.config.verbose:
                print(f"\nBenchmarking {name}...")
            all_results[name] = self.run_all(
                model, input_generator, batch_size, seq_length
            )

        return all_results

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all benchmark results."""
        summary = {
            "num_benchmarks": len(self.results),
            "by_type": {},
        }

        for result in self.results:
            btype = result.benchmark_type
            if btype not in summary["by_type"]:
                summary["by_type"][btype] = []
            summary["by_type"][btype].append(result.to_dict())

        return summary

    def save_results(self, filename: str | None = None):
        """Save all results to file."""
        if filename is None:
            filename = f"{self.config.results_dir}/benchmark_results.json"

        with open(filename, "w") as f:
            json.dump(self.get_summary(), f, indent=2)

    def reset(self):
        """Clear stored results."""
        self.results.clear()


def create_simple_input_generator(vocab_size: int = 32000, device: str = "cpu"):
    """
    Create a simple input generator for benchmarking.

    Args:
        vocab_size: Vocabulary size
        device: Device for tensors

    Returns:
        Input generator function
    """

    def generator(batch_size: int, seq_length: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.randint(
                0, vocab_size, (batch_size, seq_length), device=device
            ),
        }

    return generator
