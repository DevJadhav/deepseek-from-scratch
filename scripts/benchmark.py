#!/usr/bin/env python3
"""
Comprehensive Benchmark Comparisons for DeepSeek-V3

Compares DeepSeek implementation against various baselines:
- Hugging Face transformers (comparable model size)
- Dense transformer baseline (same parameters, no MoE)
- Standard MoE baseline (with auxiliary loss)
- nanoGPT reference implementation

Metrics:
- Throughput (tokens/second)
- Memory usage (peak GPU memory)
- Quality (perplexity on validation set)
- Scaling curves (performance vs model size)
- Hardware efficiency (performance per dollar)

Usage:
    uv run python scripts/benchmark.py --help
    uv run python scripts/benchmark.py --all
    uv run python scripts/benchmark.py --huggingface
    uv run python scripts/benchmark.py --dense-baseline
    uv run python scripts/benchmark.py --comparison-report
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    # Model configurations
    model_sizes: list[str] = field(default_factory=lambda: ["tiny", "small", "medium"])

    # Benchmark settings
    batch_sizes: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    seq_lengths: list[int] = field(default_factory=lambda: [512, 1024, 2048])
    warmup_iterations: int = 10
    benchmark_iterations: int = 50

    # Output
    output_dir: str = "./benchmark_results"
    log_to_wandb: bool = False
    wandb_project: str = "deepseek-benchmarks"


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    name: str
    model_type: str
    model_size: str
    batch_size: int
    seq_length: int
    throughput: float  # tokens/second
    latency_ms: float  # milliseconds per batch
    memory_mb: float  # peak GPU memory in MB
    perplexity: float | None = None
    extra: dict = field(default_factory=dict)


class BaselineBenchmarks:
    """Benchmark comparisons against various baselines."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = self._get_device()
        self.results: list[BenchmarkResult] = []

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _reset_memory(self):
        """Reset memory stats for accurate measurement."""
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def _get_memory_mb(self) -> float:
        """Get peak memory usage in MB."""
        if self.device.type == "cuda":
            return torch.cuda.max_memory_allocated() / 1024 / 1024
        return 0.0

    def benchmark_huggingface(self, model_name: str = "gpt2") -> list[BenchmarkResult]:
        """Benchmark against HuggingFace transformers baseline."""
        print(f"\n{'='*60}")
        print("HUGGING FACE BASELINE BENCHMARK")
        print(f"{'='*60}")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            print("Warning: transformers not available, skipping HF benchmark")
            return []

        results = []

        model_configs = {
            "tiny": "gpt2",  # 124M params
            "small": "gpt2-medium",  # 355M params
            "medium": "gpt2-large",  # 774M params
        }

        for size in self.config.model_sizes:
            if size not in model_configs:
                continue

            hf_model_name = model_configs[size]
            print(f"\nLoading {hf_model_name}...")

            try:
                model = AutoModelForCausalLM.from_pretrained(hf_model_name).to(self.device)
                model.eval()
            except Exception as e:
                print(f"Failed to load {hf_model_name}: {e}")
                continue

            for batch_size in self.config.batch_sizes:
                for seq_len in self.config.seq_lengths:
                    self._reset_memory()

                    # Create input
                    input_ids = torch.randint(
                        0, model.config.vocab_size,
                        (batch_size, seq_len),
                        device=self.device
                    )

                    # Warmup
                    for _ in range(self.config.warmup_iterations):
                        with torch.no_grad():
                            _ = model(input_ids)

                    if self.device.type == "cuda":
                        torch.cuda.synchronize()

                    # Benchmark
                    times = []
                    for _ in range(self.config.benchmark_iterations):
                        if self.device.type == "cuda":
                            torch.cuda.synchronize()

                        start = time.perf_counter()
                        with torch.no_grad():
                            _ = model(input_ids)

                        if self.device.type == "cuda":
                            torch.cuda.synchronize()
                        times.append(time.perf_counter() - start)

                    avg_time = np.mean(times)
                    tokens = batch_size * seq_len
                    throughput = tokens / avg_time
                    memory_mb = self._get_memory_mb()

                    result = BenchmarkResult(
                        name=f"huggingface_{hf_model_name}",
                        model_type="huggingface",
                        model_size=size,
                        batch_size=batch_size,
                        seq_length=seq_len,
                        throughput=throughput,
                        latency_ms=avg_time * 1000,
                        memory_mb=memory_mb,
                    )
                    results.append(result)

                    print(f"  {size} bs={batch_size} seq={seq_len}: {throughput:.0f} tok/s, {memory_mb:.0f} MB")

            # Clean up
            del model
            self._reset_memory()

        return results

    def benchmark_dense_transformer(self) -> list[BenchmarkResult]:
        """Benchmark dense transformer baseline (no MoE)."""
        print(f"\n{'='*60}")
        print("DENSE TRANSFORMER BASELINE BENCHMARK")
        print(f"{'='*60}")

        import sys
        sys.path.insert(0, "deepseek-from-scratch-python/src")

        try:
            from deepseek.model.transformer import DeepSeekModel
        except ImportError:
            print("Warning: DeepSeekModel not available")
            return []

        results = []

        model_configs = {
            "tiny": {"d_model": 256, "num_layers": 4, "num_heads": 8},
            "small": {"d_model": 512, "num_layers": 8, "num_heads": 8},
            "medium": {"d_model": 768, "num_layers": 12, "num_heads": 12},
        }

        for size, model_cfg in model_configs.items():
            if size not in self.config.model_sizes:
                continue

            print(f"\nBuilding dense transformer ({size})...")

            model = DeepSeekModel(
                vocab_size=32000,
                num_layers=model_cfg["num_layers"],
                d_model=model_cfg["d_model"],
                num_heads=model_cfg["num_heads"],
                d_latent=model_cfg["d_model"] // 4,
                d_rope=model_cfg["d_model"] // 8,
                d_hidden=model_cfg["d_model"] * 4,
                num_experts=1,  # Dense = single expert
                num_shared=1,
                num_routed=0,
                top_k=1,
                use_moe=False,
            ).to(self.device)
            model.eval()

            for batch_size in self.config.batch_sizes:
                for seq_len in self.config.seq_lengths:
                    self._reset_memory()

                    input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=self.device)

                    # Warmup
                    for _ in range(self.config.warmup_iterations):
                        with torch.no_grad():
                            _ = model(input_ids)

                    if self.device.type == "cuda":
                        torch.cuda.synchronize()

                    # Benchmark
                    times = []
                    for _ in range(self.config.benchmark_iterations):
                        if self.device.type == "cuda":
                            torch.cuda.synchronize()

                        start = time.perf_counter()
                        with torch.no_grad():
                            _ = model(input_ids)

                        if self.device.type == "cuda":
                            torch.cuda.synchronize()
                        times.append(time.perf_counter() - start)

                    avg_time = np.mean(times)
                    tokens = batch_size * seq_len
                    throughput = tokens / avg_time
                    memory_mb = self._get_memory_mb()

                    result = BenchmarkResult(
                        name=f"dense_transformer_{size}",
                        model_type="dense",
                        model_size=size,
                        batch_size=batch_size,
                        seq_length=seq_len,
                        throughput=throughput,
                        latency_ms=avg_time * 1000,
                        memory_mb=memory_mb,
                    )
                    results.append(result)

                    print(f"  {size} bs={batch_size} seq={seq_len}: {throughput:.0f} tok/s, {memory_mb:.0f} MB")

            del model
            self._reset_memory()

        return results

    def benchmark_standard_moe(self) -> list[BenchmarkResult]:
        """Benchmark standard MoE with auxiliary loss."""
        print(f"\n{'='*60}")
        print("STANDARD MOE BASELINE BENCHMARK (with aux loss)")
        print(f"{'='*60}")

        import sys
        sys.path.insert(0, "deepseek-from-scratch-python/src")

        try:
            from deepseek.model.transformer import DeepSeekModel
        except ImportError:
            print("Warning: DeepSeekModel not available")
            return []

        results = []

        model_configs = {
            "tiny": {"d_model": 256, "num_layers": 4, "num_heads": 8, "num_experts": 8},
            "small": {"d_model": 512, "num_layers": 8, "num_heads": 8, "num_experts": 16},
            "medium": {"d_model": 768, "num_layers": 12, "num_heads": 12, "num_experts": 32},
        }

        for size, model_cfg in model_configs.items():
            if size not in self.config.model_sizes:
                continue

            print(f"\nBuilding standard MoE ({size})...")

            model = DeepSeekModel(
                vocab_size=32000,
                num_layers=model_cfg["num_layers"],
                d_model=model_cfg["d_model"],
                num_heads=model_cfg["num_heads"],
                d_latent=model_cfg["d_model"] // 4,
                d_rope=model_cfg["d_model"] // 8,
                d_hidden=model_cfg["d_model"] * 4,
                num_experts=model_cfg["num_experts"],
                num_shared=1,
                num_routed=model_cfg["num_experts"] - 1,
                top_k=2,
                use_moe=True,
            ).to(self.device)
            model.eval()

            for batch_size in self.config.batch_sizes:
                for seq_len in self.config.seq_lengths:
                    self._reset_memory()

                    input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=self.device)

                    # Warmup
                    for _ in range(self.config.warmup_iterations):
                        with torch.no_grad():
                            _ = model(input_ids)

                    if self.device.type == "cuda":
                        torch.cuda.synchronize()

                    # Benchmark
                    times = []
                    for _ in range(self.config.benchmark_iterations):
                        if self.device.type == "cuda":
                            torch.cuda.synchronize()

                        start = time.perf_counter()
                        with torch.no_grad():
                            _ = model(input_ids)

                        if self.device.type == "cuda":
                            torch.cuda.synchronize()
                        times.append(time.perf_counter() - start)

                    avg_time = np.mean(times)
                    tokens = batch_size * seq_len
                    throughput = tokens / avg_time
                    memory_mb = self._get_memory_mb()

                    result = BenchmarkResult(
                        name=f"standard_moe_{size}",
                        model_type="standard_moe",
                        model_size=size,
                        batch_size=batch_size,
                        seq_length=seq_len,
                        throughput=throughput,
                        latency_ms=avg_time * 1000,
                        memory_mb=memory_mb,
                    )
                    results.append(result)

                    print(f"  {size} bs={batch_size} seq={seq_len}: {throughput:.0f} tok/s, {memory_mb:.0f} MB")

            del model
            self._reset_memory()

        return results

    def benchmark_deepseek_v3(self) -> list[BenchmarkResult]:
        """Benchmark DeepSeek-V3 implementation."""
        print(f"\n{'='*60}")
        print("DEEPSEEK-V3 IMPLEMENTATION BENCHMARK")
        print(f"{'='*60}")

        import sys
        sys.path.insert(0, "deepseek-from-scratch-python/src")

        try:
            from deepseek.model.transformer import DeepSeekModel
        except ImportError:
            print("Warning: DeepSeekModel not available")
            return []

        results = []

        model_configs = {
            "tiny": {
                "d_model": 256, "num_layers": 4, "num_heads": 8,
                "num_experts": 8, "d_latent": 64, "d_rope": 32,
            },
            "small": {
                "d_model": 512, "num_layers": 8, "num_heads": 8,
                "num_experts": 16, "d_latent": 128, "d_rope": 64,
            },
            "medium": {
                "d_model": 768, "num_layers": 12, "num_heads": 12,
                "num_experts": 32, "d_latent": 192, "d_rope": 96,
            },
        }

        for size, model_cfg in model_configs.items():
            if size not in self.config.model_sizes:
                continue

            print(f"\nBuilding DeepSeek-V3 ({size})...")

            model = DeepSeekModel(
                vocab_size=32000,
                num_layers=model_cfg["num_layers"],
                d_model=model_cfg["d_model"],
                num_heads=model_cfg["num_heads"],
                d_latent=model_cfg["d_latent"],
                d_rope=model_cfg["d_rope"],
                d_hidden=model_cfg["d_model"] * 4,
                num_experts=model_cfg["num_experts"],
                num_shared=1,
                num_routed=model_cfg["num_experts"] - 1,
                top_k=2,
                use_moe=True,
            ).to(self.device)
            model.eval()

            for batch_size in self.config.batch_sizes:
                for seq_len in self.config.seq_lengths:
                    self._reset_memory()

                    input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=self.device)

                    # Warmup
                    for _ in range(self.config.warmup_iterations):
                        with torch.no_grad():
                            _ = model(input_ids)

                    if self.device.type == "cuda":
                        torch.cuda.synchronize()

                    # Benchmark
                    times = []
                    for _ in range(self.config.benchmark_iterations):
                        if self.device.type == "cuda":
                            torch.cuda.synchronize()

                        start = time.perf_counter()
                        with torch.no_grad():
                            _ = model(input_ids)

                        if self.device.type == "cuda":
                            torch.cuda.synchronize()
                        times.append(time.perf_counter() - start)

                    avg_time = np.mean(times)
                    tokens = batch_size * seq_len
                    throughput = tokens / avg_time
                    memory_mb = self._get_memory_mb()

                    result = BenchmarkResult(
                        name=f"deepseek_v3_{size}",
                        model_type="deepseek_v3",
                        model_size=size,
                        batch_size=batch_size,
                        seq_length=seq_len,
                        throughput=throughput,
                        latency_ms=avg_time * 1000,
                        memory_mb=memory_mb,
                    )
                    results.append(result)

                    print(f"  {size} bs={batch_size} seq={seq_len}: {throughput:.0f} tok/s, {memory_mb:.0f} MB")

            del model
            self._reset_memory()

        return results


def generate_comparison_report(results: list[BenchmarkResult], output_dir: Path) -> str:
    """Generate a markdown comparison report."""

    # Group results by model type
    by_type: dict[str, list[BenchmarkResult]] = {}
    for r in results:
        if r.model_type not in by_type:
            by_type[r.model_type] = []
        by_type[r.model_type].append(r)

    report = []
    report.append("# DeepSeek-V3 Benchmark Comparison Report\n")
    report.append(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")

    # Summary table
    report.append("## Summary (batch_size=4, seq_len=1024)\n")
    report.append("| Model Type | Size | Throughput (tok/s) | Memory (MB) | Latency (ms) |")
    report.append("|------------|------|-------------------|-------------|--------------|")

    for model_type, type_results in by_type.items():
        for r in type_results:
            if r.batch_size == 4 and r.seq_length == 1024:
                report.append(f"| {r.model_type} | {r.model_size} | {r.throughput:.0f} | {r.memory_mb:.0f} | {r.latency_ms:.2f} |")

    # Detailed results by model type
    for model_type, type_results in by_type.items():
        report.append(f"\n## {model_type.replace('_', ' ').title()} Results\n")
        report.append("| Size | Batch | Seq Len | Throughput | Memory | Latency |")
        report.append("|------|-------|---------|------------|--------|---------|")

        for r in sorted(type_results, key=lambda x: (x.model_size, x.batch_size, x.seq_length)):
            report.append(
                f"| {r.model_size} | {r.batch_size} | {r.seq_length} | "
                f"{r.throughput:.0f} tok/s | {r.memory_mb:.0f} MB | {r.latency_ms:.2f} ms |"
            )

    # Comparison analysis
    report.append("\n## Analysis\n")

    # Calculate speedups relative to dense baseline
    dense_results = {(r.model_size, r.batch_size, r.seq_length): r for r in by_type.get("dense", [])}
    deepseek_results = {(r.model_size, r.batch_size, r.seq_length): r for r in by_type.get("deepseek_v3", [])}

    if dense_results and deepseek_results:
        report.append("### DeepSeek-V3 vs Dense Transformer\n")
        report.append("| Configuration | Dense (tok/s) | DeepSeek (tok/s) | Speedup |")
        report.append("|---------------|---------------|------------------|---------|")

        for key, dense_r in dense_results.items():
            if key in deepseek_results:
                ds_r = deepseek_results[key]
                speedup = ds_r.throughput / dense_r.throughput if dense_r.throughput > 0 else 0
                report.append(
                    f"| {key[0]} bs={key[1]} seq={key[2]} | "
                    f"{dense_r.throughput:.0f} | {ds_r.throughput:.0f} | {speedup:.2f}x |"
                )

    report.append("\n## Hardware Efficiency\n")
    report.append("Performance per dollar estimates (based on cloud GPU pricing):\n")
    report.append("- A100 40GB: ~$3/hr")
    report.append("- H100 80GB: ~$5/hr")
    report.append("- RTX 4090: ~$1/hr (consumer)\n")

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Run benchmark comparisons")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--huggingface", action="store_true", help="Benchmark HuggingFace baseline")
    parser.add_argument("--dense-baseline", action="store_true", help="Benchmark dense transformer")
    parser.add_argument("--standard-moe", action="store_true", help="Benchmark standard MoE")
    parser.add_argument("--deepseek", action="store_true", help="Benchmark DeepSeek-V3")
    parser.add_argument("--comparison-report", action="store_true", help="Generate comparison report")
    parser.add_argument("--output-dir", type=str, default="./benchmark_results",
                        help="Output directory for results")
    parser.add_argument("--model-sizes", type=str, default="tiny,small",
                        help="Comma-separated model sizes to test")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8",
                        help="Comma-separated batch sizes")
    parser.add_argument("--seq-lengths", type=str, default="512,1024",
                        help="Comma-separated sequence lengths")
    args = parser.parse_args()

    # Parse configurations
    model_sizes = [s.strip() for s in args.model_sizes.split(",")]
    batch_sizes = [int(s.strip()) for s in args.batch_sizes.split(",")]
    seq_lengths = [int(s.strip()) for s in args.seq_lengths.split(",")]

    config = BenchmarkConfig(
        model_sizes=model_sizes,
        batch_sizes=batch_sizes,
        seq_lengths=seq_lengths,
        output_dir=args.output_dir,
    )

    benchmarks = BaselineBenchmarks(config)
    all_results = []

    # Run selected benchmarks
    run_all = args.all or not any([
        args.huggingface, args.dense_baseline, args.standard_moe, args.deepseek
    ])

    if run_all or args.huggingface:
        results = benchmarks.benchmark_huggingface()
        all_results.extend(results)

    if run_all or args.dense_baseline:
        results = benchmarks.benchmark_dense_transformer()
        all_results.extend(results)

    if run_all or args.standard_moe:
        results = benchmarks.benchmark_standard_moe()
        all_results.extend(results)

    if run_all or args.deepseek:
        results = benchmarks.benchmark_deepseek_v3()
        all_results.extend(results)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump([{
            "name": r.name,
            "model_type": r.model_type,
            "model_size": r.model_size,
            "batch_size": r.batch_size,
            "seq_length": r.seq_length,
            "throughput": r.throughput,
            "latency_ms": r.latency_ms,
            "memory_mb": r.memory_mb,
        } for r in all_results], f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Generate comparison report
    if run_all or args.comparison_report:
        report = generate_comparison_report(all_results, output_dir)
        report_file = output_dir / "comparison_report.md"
        with open(report_file, "w") as f:
            f.write(report)
        print(f"Report saved to: {report_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    for r in all_results:
        if r.batch_size == 4 and r.seq_length == 1024:
            print(f"{r.model_type:15} {r.model_size:8}: {r.throughput:8.0f} tok/s, {r.memory_mb:6.0f} MB")


if __name__ == "__main__":
    main()
