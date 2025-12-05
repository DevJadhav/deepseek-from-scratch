#!/usr/bin/env python3
"""
GPU Optimization Benchmark Script for DeepSeek

This script benchmarks core GPU optimizations:
- Flash Attention vs Vanilla Attention
- Compiled vs Uncompiled models
- Mixed Precision (BF16/FP16/FP32)
- Gradient Checkpointing memory savings

Usage:
    uv run python scripts/benchmark_gpu_optimization.py --help
    uv run python scripts/benchmark_gpu_optimization.py --all
    uv run python scripts/benchmark_gpu_optimization.py --attention
    uv run python scripts/benchmark_gpu_optimization.py --compile
    uv run python scripts/benchmark_gpu_optimization.py --precision
"""

import argparse
import sys
import time
from typing import Optional
from dataclasses import dataclass
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent directory for imports
sys.path.insert(0, "deepseek-from-scratch-python/src")

from deepseek.model.attention import (
    AttentionBackend,
    FlashAttentionConfig,
    MultiQueryAttention,
    GroupedQueryAttention,
    detect_flash_attention_version,
    get_optimal_attention_backend,
)
from deepseek.model.transformer import (
    DeepSeekModel,
    GradientCheckpointConfig,
)
from deepseek.training.optimization import (
    CompileMode,
    CompileConfig,
    compile_model,
    PrecisionMode,
    MixedPrecisionConfig,
    MixedPrecisionTrainer,
    get_optimal_precision,
    MemoryProfiler,
)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    time_ms: float
    memory_mb: float
    throughput: float  # tokens/second
    extra: Optional[dict] = None


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def benchmark_attention_backends(
    batch_size: int = 4,
    seq_len: int = 1024,
    d_model: int = 512,
    num_heads: int = 8,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
) -> list[BenchmarkResult]:
    """
    Benchmark different attention backends.
    """
    device = get_device()
    print(f"\n{'='*60}")
    print("ATTENTION BACKEND BENCHMARK")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Batch: {batch_size}, Seq: {seq_len}, Model: {d_model}, Heads: {num_heads}")
    print(f"Iterations: {num_iterations} (warmup: {warmup_iterations})")
    
    # Check Flash Attention availability
    fa_version = detect_flash_attention_version()
    optimal_backend = get_optimal_attention_backend()
    print(f"\nFlash Attention Version: {fa_version}")
    print(f"Optimal Backend: {optimal_backend}")
    
    results = []
    backends_to_test = [AttentionBackend.MATH]
    
    if device.type == "cuda":
        backends_to_test.extend([AttentionBackend.MEMORY_EFFICIENT])
        if fa_version:
            backends_to_test.append(AttentionBackend.FLASH)
    
    for backend in backends_to_test:
        print(f"\n--- Testing {backend.value} backend ---")
        
        config = FlashAttentionConfig(backend=backend)
        model = MultiQueryAttention(d_model, num_heads, attention_config=config).to(device)
        
        # Create input
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        
        # Warmup
        for _ in range(warmup_iterations):
            with torch.no_grad():
                _ = model(x)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        
        # Benchmark
        times = []
        for _ in range(num_iterations):
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(x)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        
        avg_time = sum(times) / len(times) * 1000  # ms
        tokens = batch_size * seq_len
        throughput = tokens / (avg_time / 1000)
        
        memory_mb = 0
        if device.type == "cuda":
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        print(f"  Avg Time: {avg_time:.2f} ms")
        print(f"  Memory: {memory_mb:.1f} MB")
        print(f"  Throughput: {throughput:.0f} tokens/s")
        
        results.append(BenchmarkResult(
            name=f"attention_{backend.value}",
            time_ms=avg_time,
            memory_mb=memory_mb,
            throughput=throughput,
        ))
    
    return results


def benchmark_torch_compile(
    batch_size: int = 4,
    seq_len: int = 256,
    vocab_size: int = 32000,
    d_model: int = 256,
    num_layers: int = 4,
    num_iterations: int = 50,
    warmup_iterations: int = 10,
) -> list[BenchmarkResult]:
    """
    Benchmark compiled vs uncompiled models.
    """
    device = get_device()
    print(f"\n{'='*60}")
    print("TORCH.COMPILE BENCHMARK")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Batch: {batch_size}, Seq: {seq_len}, Layers: {num_layers}")
    
    if device.type != "cuda":
        print("⚠️  torch.compile is most effective on CUDA devices")
    
    results = []
    
    # Test configurations
    configs = [
        ("uncompiled", CompileConfig(mode=CompileMode.DISABLED)),
        ("compiled_reduce_overhead", CompileConfig(mode=CompileMode.REDUCE_OVERHEAD)),
    ]
    
    if device.type == "cuda":
        configs.append(("compiled_max_autotune", CompileConfig(mode=CompileMode.MAX_AUTOTUNE)))
    
    for name, compile_config in configs:
        print(f"\n--- Testing {name} ---")
        
        # Create model
        model = DeepSeekModel(
            vocab_size=vocab_size,
            num_layers=num_layers,
            d_model=d_model,
            num_heads=8,
            d_latent=64,
            d_rope=32,
            d_hidden=d_model * 4,
            num_experts=4,
            num_shared=1,
            num_routed=4,
            top_k=2,
            use_moe=False,  # Simpler for benchmark
        ).to(device)
        
        # Compile if enabled
        if compile_config.mode != CompileMode.DISABLED:
            try:
                model = compile_model(model, compile_config)
                print(f"  Model compiled with {compile_config.mode.value}")
            except Exception as e:
                print(f"  Compilation failed: {e}")
                continue
        
        # Create input
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        # Warmup (more for compiled models)
        warmup = warmup_iterations * 3 if compile_config.mode != CompileMode.DISABLED else warmup_iterations
        for _ in range(warmup):
            with torch.no_grad():
                _ = model(input_ids)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        
        # Benchmark
        times = []
        for _ in range(num_iterations):
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(input_ids)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        
        avg_time = sum(times) / len(times) * 1000
        tokens = batch_size * seq_len
        throughput = tokens / (avg_time / 1000)
        
        memory_mb = 0
        if device.type == "cuda":
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        print(f"  Avg Time: {avg_time:.2f} ms")
        print(f"  Memory: {memory_mb:.1f} MB")
        print(f"  Throughput: {throughput:.0f} tokens/s")
        
        results.append(BenchmarkResult(
            name=f"compile_{name}",
            time_ms=avg_time,
            memory_mb=memory_mb,
            throughput=throughput,
        ))
    
    return results


def benchmark_mixed_precision(
    batch_size: int = 4,
    seq_len: int = 256,
    vocab_size: int = 32000,
    d_model: int = 256,
    num_layers: int = 4,
    num_iterations: int = 50,
    warmup_iterations: int = 10,
) -> list[BenchmarkResult]:
    """
    Benchmark different precision modes.
    """
    device = get_device()
    print(f"\n{'='*60}")
    print("MIXED PRECISION BENCHMARK")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Batch: {batch_size}, Seq: {seq_len}, Layers: {num_layers}")
    
    optimal = get_optimal_precision()
    print(f"Optimal Precision: {optimal.value}")
    
    if device.type != "cuda":
        print("⚠️  Mixed precision training is most effective on CUDA devices")
        return []
    
    results = []
    
    # Test configurations
    precision_modes = [PrecisionMode.FP32, PrecisionMode.FP16]
    if torch.cuda.get_device_capability()[0] >= 8:
        precision_modes.append(PrecisionMode.BF16)
    
    for mode in precision_modes:
        print(f"\n--- Testing {mode.value} ---")
        
        # Create model in FP32 first
        model = DeepSeekModel(
            vocab_size=vocab_size,
            num_layers=num_layers,
            d_model=d_model,
            num_heads=8,
            d_latent=64,
            d_rope=32,
            d_hidden=d_model * 4,
            num_experts=4,
            num_shared=1,
            num_routed=4,
            top_k=2,
            use_moe=False,
        ).to(device)
        
        # Create mixed precision trainer
        mp_config = MixedPrecisionConfig(mode=mode)
        mp_trainer = MixedPrecisionTrainer(mp_config)
        
        # Create input and target
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Warmup
        for _ in range(warmup_iterations):
            optimizer.zero_grad()
            with mp_trainer.autocast_context():
                logits = model(input_ids)
                loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
            mp_trainer.scale_loss(loss).backward()
            mp_trainer.optimizer_step(optimizer)
        
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
        # Benchmark
        times = []
        for _ in range(num_iterations):
            optimizer.zero_grad()
            torch.cuda.synchronize()
            
            start = time.perf_counter()
            with mp_trainer.autocast_context():
                logits = model(input_ids)
                loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
            mp_trainer.scale_loss(loss).backward()
            mp_trainer.optimizer_step(optimizer)
            
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        
        avg_time = sum(times) / len(times) * 1000
        tokens = batch_size * seq_len
        throughput = tokens / (avg_time / 1000)
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        print(f"  Avg Time: {avg_time:.2f} ms")
        print(f"  Memory: {memory_mb:.1f} MB")
        print(f"  Throughput: {throughput:.0f} tokens/s")
        
        results.append(BenchmarkResult(
            name=f"precision_{mode.value}",
            time_ms=avg_time,
            memory_mb=memory_mb,
            throughput=throughput,
        ))
    
    return results


def benchmark_gradient_checkpointing(
    batch_size: int = 4,
    seq_len: int = 512,
    vocab_size: int = 32000,
    d_model: int = 512,
    num_layers: int = 12,
    num_iterations: int = 20,
    warmup_iterations: int = 5,
) -> list[BenchmarkResult]:
    """
    Benchmark gradient checkpointing memory savings.
    """
    device = get_device()
    print(f"\n{'='*60}")
    print("GRADIENT CHECKPOINTING BENCHMARK")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Batch: {batch_size}, Seq: {seq_len}, Layers: {num_layers}")
    
    if device.type != "cuda":
        print("⚠️  Gradient checkpointing memory savings only measurable on CUDA")
        return []
    
    results = []
    
    # Test configurations
    configs = [
        ("no_checkpointing", GradientCheckpointConfig(enabled=False)),
        ("checkpoint_all", GradientCheckpointConfig(enabled=True, checkpoint_every_n_layers=1)),
        ("checkpoint_every_2", GradientCheckpointConfig(enabled=True, checkpoint_every_n_layers=2)),
    ]
    
    for name, ckpt_config in configs:
        print(f"\n--- Testing {name} ---")
        
        # Clear memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        try:
            model = DeepSeekModel(
                vocab_size=vocab_size,
                num_layers=num_layers,
                d_model=d_model,
                num_heads=8,
                d_latent=128,
                d_rope=32,
                d_hidden=d_model * 4,
                num_experts=4,
                num_shared=1,
                num_routed=4,
                top_k=2,
                use_moe=False,
                checkpoint_config=ckpt_config,
            ).to(device)
            
            # Create input and target
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            
            # Warmup
            for _ in range(warmup_iterations):
                optimizer.zero_grad()
                logits = model(input_ids)
                loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
                loss.backward()
                optimizer.step()
            
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            
            # Benchmark
            times = []
            for _ in range(num_iterations):
                optimizer.zero_grad()
                torch.cuda.synchronize()
                
                start = time.perf_counter()
                logits = model(input_ids)
                loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
                loss.backward()
                optimizer.step()
                
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
            
            avg_time = sum(times) / len(times) * 1000
            tokens = batch_size * seq_len
            throughput = tokens / (avg_time / 1000)
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            print(f"  Avg Time: {avg_time:.2f} ms")
            print(f"  Peak Memory: {memory_mb:.1f} MB")
            print(f"  Throughput: {throughput:.0f} tokens/s")
            
            results.append(BenchmarkResult(
                name=f"checkpoint_{name}",
                time_ms=avg_time,
                memory_mb=memory_mb,
                throughput=throughput,
            ))
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  ❌ Out of Memory")
                results.append(BenchmarkResult(
                    name=f"checkpoint_{name}",
                    time_ms=float('inf'),
                    memory_mb=float('inf'),
                    throughput=0,
                ))
            else:
                raise
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    return results


def print_summary(all_results: list[BenchmarkResult]):
    """Print benchmark summary."""
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"{'Name':<35} {'Time (ms)':<12} {'Memory (MB)':<12} {'Throughput':<12}")
    print("-" * 71)
    
    for result in all_results:
        time_str = f"{result.time_ms:.2f}" if result.time_ms != float('inf') else "OOM"
        mem_str = f"{result.memory_mb:.1f}" if result.memory_mb != float('inf') else "OOM"
        throughput_str = f"{result.throughput:.0f}" if result.throughput > 0 else "N/A"
        print(f"{result.name:<35} {time_str:<12} {mem_str:<12} {throughput_str:<12}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Benchmark Script")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--attention", action="store_true", help="Benchmark attention backends")
    parser.add_argument("--compile", action="store_true", help="Benchmark torch.compile")
    parser.add_argument("--precision", action="store_true", help="Benchmark mixed precision")
    parser.add_argument("--checkpoint", action="store_true", help="Benchmark gradient checkpointing")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    
    args = parser.parse_args()
    
    # Default to all if nothing specified
    if not any([args.all, args.attention, args.compile, args.precision, args.checkpoint]):
        args.all = True
    
    all_results = []
    
    print("\n" + "=" * 60)
    print("DeepSeek Phase 1 Benchmark")
    print("=" * 60)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name()}")
        print(f"CUDA Capability: {torch.cuda.get_device_capability()}")
    
    if args.all or args.attention:
        results = benchmark_attention_backends()
        all_results.extend(results)
    
    if args.all or args.compile:
        results = benchmark_torch_compile()
        all_results.extend(results)
    
    if args.all or args.precision:
        results = benchmark_mixed_precision()
        all_results.extend(results)
    
    if args.all or args.checkpoint:
        results = benchmark_gradient_checkpointing()
        all_results.extend(results)
    
    print_summary(all_results)
    
    if args.output:
        output_data = [
            {
                "name": r.name,
                "time_ms": r.time_ms if r.time_ms != float('inf') else None,
                "memory_mb": r.memory_mb if r.memory_mb != float('inf') else None,
                "throughput": r.throughput if r.throughput > 0 else None,
            }
            for r in all_results
        ]
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
