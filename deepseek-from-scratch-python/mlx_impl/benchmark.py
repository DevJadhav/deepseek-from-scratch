"""
Comprehensive benchmark suite for DeepSeek from Scratch (MLX)
Uses time.perf_counter() for precise timing with multiple warmup and measurement iterations.
"""

import mlx.core as mx
import mlx.nn as nn
import time
import gc
from dataclasses import dataclass
from typing import Callable, Any

from attention import MultiQueryAttention, GroupedQueryAttention, MultiHeadLatentAttention
from moe import DeepSeekMoE
from mtp import MTPModel
from grpo import GRPOTrainer
from r1 import ReasoningModel
from distillation import kd_loss_kl, kd_loss_mse, kd_loss_jsd
from dpo import DPOTrainer, DPOConfig
from reward_model import RewardModel, RewardConfig


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarks."""
    batch_size: int = 4
    seq_len: int = 64
    d_model: int = 512
    warmup_iters: int = 5
    measure_iters: int = 20


def benchmark_function(
    fn: Callable[[], Any],
    warmup_iters: int = 5,
    measure_iters: int = 20,
) -> tuple[float, float]:
    """
    Benchmark a function with proper warmup and synchronization.
    Returns (mean_ms, std_ms).
    """
    # Warmup
    for _ in range(warmup_iters):
        result = fn()
        mx.eval(result)  # Force evaluation
    
    # Collect garbage before measuring
    gc.collect()
    
    # Measure
    times = []
    for _ in range(measure_iters):
        mx.synchronize()  # Ensure all previous ops complete
        start = time.perf_counter()
        result = fn()
        mx.eval(result)  # Force evaluation
        mx.synchronize()  # Wait for completion
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
    return mean_time, std_time


def run_attention_benchmarks(config: BenchmarkConfig) -> dict:
    """Run benchmarks for all attention mechanisms."""
    results = {}
    x = mx.random.normal((config.batch_size, config.seq_len, config.d_model))
    mx.eval(x)
    
    print("\n" + "=" * 60)
    print("Attention Mechanism Benchmarks (MLX)")
    print("=" * 60)
    print(f"Config: batch={config.batch_size}, seq_len={config.seq_len}, "
          f"d_model={config.d_model}")
    print(f"Warmup: {config.warmup_iters}, Measure: {config.measure_iters} iterations")
    print("-" * 60)
    
    # MQA
    mqa = MultiQueryAttention(d_model=config.d_model, num_heads=8)
    mean, std = benchmark_function(
        lambda: mqa(x), config.warmup_iters, config.measure_iters
    )
    results["MQA"] = (mean, std)
    print(f"MQA:          {mean:7.2f}ms ± {std:.2f}ms")
    
    # GQA
    gqa = GroupedQueryAttention(
        d_model=config.d_model, num_heads=32, num_groups=4
    )
    mean, std = benchmark_function(
        lambda: gqa(x), config.warmup_iters, config.measure_iters
    )
    results["GQA"] = (mean, std)
    print(f"GQA:          {mean:7.2f}ms ± {std:.2f}ms")
    
    # MLA
    mla = MultiHeadLatentAttention(
        d_model=config.d_model, num_heads=8, d_latent=128, d_rope=64
    )
    mean, std = benchmark_function(
        lambda: mla(x), config.warmup_iters, config.measure_iters
    )
    results["MLA"] = (mean, std)
    print(f"MLA:          {mean:7.2f}ms ± {std:.2f}ms")
    
    return results


def run_moe_benchmarks(config: BenchmarkConfig) -> dict:
    """Run benchmarks for MoE variants."""
    results = {}
    x = mx.random.normal((config.batch_size, config.seq_len, config.d_model))
    mx.eval(x)
    
    print("\n" + "=" * 60)
    print("Mixture of Experts Benchmarks (MLX)")
    print("=" * 60)
    print(f"Config: batch={config.batch_size}, seq_len={config.seq_len}, "
          f"d_model={config.d_model}")
    print("-" * 60)
    
    # DeepSeek MoE
    ds_moe = DeepSeekMoE(
        d_model=config.d_model, d_hidden=1024, 
        num_experts=10, num_shared=2, num_routed=8, top_k=2
    )
    mean, std = benchmark_function(
        lambda: ds_moe(x), config.warmup_iters, config.measure_iters
    )
    results["DeepSeek MoE"] = (mean, std)
    print(f"DeepSeek MoE:  {mean:7.2f}ms ± {std:.2f}ms")
    
    return results


def run_mtp_benchmarks(config: BenchmarkConfig) -> dict:
    """Run benchmarks for Multi-Token Prediction."""
    results = {}
    input_ids = mx.random.randint(0, 1000, (config.batch_size, config.seq_len))
    mx.eval(input_ids)
    
    print("\n" + "=" * 60)
    print("Multi-Token Prediction Benchmarks (MLX)")
    print("=" * 60)
    print(f"Config: batch={config.batch_size}, seq_len={config.seq_len}, "
          f"d_model={config.d_model}")
    print("-" * 60)
    
    # MTP with 1 future prediction
    mtp = MTPModel(
        vocab_size=1000, d_model=config.d_model, num_layers=2, k_predictions=1
    )
    mean, std = benchmark_function(
        lambda: mtp(input_ids), config.warmup_iters, config.measure_iters
    )
    results["MTP"] = (mean, std)
    print(f"MTP (k=1):     {mean:7.2f}ms ± {std:.2f}ms")
    
    return results


def run_grpo_benchmarks(config: BenchmarkConfig) -> dict:
    """Run benchmarks for GRPO (Chapter 9)."""
    results = {}
    
    print("\n" + "=" * 60)
    print("GRPO (Group Relative Policy Optimization) Benchmarks (MLX)")
    print("=" * 60)
    print(f"Config: batch={config.batch_size}, seq_len={config.seq_len}, "
          f"d_model={config.d_model}")
    print("-" * 60)
    
    vocab_size = 1000
    group_size = config.batch_size
    
    # Create GRPO trainer
    grpo = GRPOTrainer(beta=0.01)
    
    # Prepare inputs
    logits = mx.random.normal((group_size, config.seq_len, vocab_size))
    input_ids = mx.random.randint(0, vocab_size, (group_size, config.seq_len))
    rewards = mx.random.normal((group_size,))
    ref_logits = mx.random.normal((group_size, config.seq_len, vocab_size))
    mx.eval(logits, input_ids, rewards, ref_logits)
    
    mean, std = benchmark_function(
        lambda: grpo.compute_loss(logits, input_ids, rewards, ref_logits),
        config.warmup_iters, config.measure_iters
    )
    results["GRPO Loss"] = (mean, std)
    print(f"GRPO Loss:     {mean:7.2f}ms ± {std:.2f}ms")
    
    return results


def run_r1_benchmarks(config: BenchmarkConfig) -> dict:
    """Run benchmarks for R1 Reasoning Model (Chapter 9)."""
    results = {}
    
    print("\n" + "=" * 60)
    print("R1 Reasoning Model Benchmarks (MLX)")
    print("=" * 60)
    print(f"Config: batch={config.batch_size}, seq_len={config.seq_len}, "
          f"d_model={config.d_model}")
    print("-" * 60)
    
    vocab_size = 1000
    
    # Create R1 model
    r1 = ReasoningModel(vocab_size=vocab_size, d_model=config.d_model)
    input_ids = mx.random.randint(0, vocab_size, (config.batch_size, config.seq_len))
    mx.eval(input_ids)
    
    # R1 Forward pass
    mean, std = benchmark_function(
        lambda: r1(input_ids),
        config.warmup_iters, config.measure_iters
    )
    results["R1 Forward"] = (mean, std)
    print(f"R1 Forward:    {mean:7.2f}ms ± {std:.2f}ms")
    
    return results


def run_distillation_benchmarks(config: BenchmarkConfig) -> dict:
    """Run benchmarks for Knowledge Distillation losses (Chapter 13)."""
    results = {}
    
    print("\n" + "=" * 60)
    print("Knowledge Distillation Benchmarks (MLX)")
    print("=" * 60)
    print(f"Config: batch={config.batch_size}, seq_len={config.seq_len}, "
          f"d_model={config.d_model}")
    print("-" * 60)
    
    vocab_size = 1000
    
    # Prepare logits
    student_logits = mx.random.normal((config.batch_size, config.seq_len, vocab_size))
    teacher_logits = mx.random.normal((config.batch_size, config.seq_len, vocab_size))
    mx.eval(student_logits, teacher_logits)
    
    # KL Divergence Loss
    mean, std = benchmark_function(
        lambda: kd_loss_kl(student_logits, teacher_logits, temperature=4.0),
        config.warmup_iters, config.measure_iters
    )
    results["KD KL Loss"] = (mean, std)
    print(f"KD KL Loss:    {mean:7.2f}ms ± {std:.2f}ms")
    
    # MSE Loss
    mean, std = benchmark_function(
        lambda: kd_loss_mse(student_logits, teacher_logits, temperature=4.0),
        config.warmup_iters, config.measure_iters
    )
    results["KD MSE Loss"] = (mean, std)
    print(f"KD MSE Loss:   {mean:7.2f}ms ± {std:.2f}ms")
    
    # JSD Loss
    mean, std = benchmark_function(
        lambda: kd_loss_jsd(student_logits, teacher_logits, temperature=4.0),
        config.warmup_iters, config.measure_iters
    )
    results["KD JSD Loss"] = (mean, std)
    print(f"KD JSD Loss:   {mean:7.2f}ms ± {std:.2f}ms")
    
    return results


def run_dpo_benchmarks(config: BenchmarkConfig) -> dict:
    """Run benchmarks for DPO loss computation (Chapter 12)."""
    results = {}
    
    print("\n" + "=" * 60)
    print("DPO (Direct Preference Optimization) Benchmarks (MLX)")
    print("=" * 60)
    print(f"Config: batch={config.batch_size}, seq_len={config.seq_len}, "
          f"d_model={config.d_model}")
    print("-" * 60)
    
    # Create DPO trainer
    dpo_config = DPOConfig(beta=0.1)
    dpo = DPOTrainer(model=None, ref_model=None, config=dpo_config)
    
    # Prepare log probabilities
    policy_chosen_logps = mx.random.normal((config.batch_size,))
    policy_rejected_logps = mx.random.normal((config.batch_size,))
    ref_chosen_logps = mx.random.normal((config.batch_size,))
    ref_rejected_logps = mx.random.normal((config.batch_size,))
    mx.eval(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps)
    
    mean, std = benchmark_function(
        lambda: dpo.compute_dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            ref_chosen_logps, ref_rejected_logps
        )[0],
        config.warmup_iters, config.measure_iters
    )
    results["DPO Loss"] = (mean, std)
    print(f"DPO Loss:      {mean:7.2f}ms ± {std:.2f}ms")
    
    return results


def run_reward_benchmarks(config: BenchmarkConfig) -> dict:
    """Run benchmarks for Reward Model (Chapter 12)."""
    results = {}
    
    print("\n" + "=" * 60)
    print("Reward Model Benchmarks (MLX)")
    print("=" * 60)
    print(f"Config: batch={config.batch_size}, seq_len={config.seq_len}, "
          f"d_model={config.d_model}")
    print("-" * 60)
    
    # Create a simple base model
    class SimpleBaseModel(nn.Module):
        def __init__(self, vocab_size, d_model):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            
        def __call__(self, input_ids):
            return {"hidden_states": self.embed(input_ids)}
    
    vocab_size = 1000
    base_model = SimpleBaseModel(vocab_size, config.d_model)
    
    # Create reward model
    reward_config = RewardConfig(hidden_size=config.d_model)
    reward_model = RewardModel(base_model, reward_config)
    
    input_ids = mx.random.randint(0, vocab_size, (config.batch_size, config.seq_len))
    attention_mask = mx.ones((config.batch_size, config.seq_len))
    mx.eval(input_ids, attention_mask)
    
    mean, std = benchmark_function(
        lambda: reward_model(input_ids, attention_mask),
        config.warmup_iters, config.measure_iters
    )
    results["Reward Forward"] = (mean, std)
    print(f"Reward Forward: {mean:6.2f}ms ± {std:.2f}ms")
    
    return results


def run_all_benchmarks():
    """Run all benchmarks and print summary."""
    config = BenchmarkConfig()
    
    # Set MLX to GPU
    mx.set_default_device(mx.gpu)
    
    print("=" * 60)
    print("DeepSeek from Scratch - MLX Benchmarks")
    print("=" * 60)
    print(f"Device: Metal GPU")
    print(f"MLX version: {mx.__version__ if hasattr(mx, '__version__') else 'N/A'}")
    
    all_results = {}
    
    # Run all benchmark suites
    all_results.update(run_attention_benchmarks(config))
    all_results.update(run_moe_benchmarks(config))
    all_results.update(run_mtp_benchmarks(config))
    all_results.update(run_grpo_benchmarks(config))
    all_results.update(run_r1_benchmarks(config))
    all_results.update(run_distillation_benchmarks(config))
    all_results.update(run_dpo_benchmarks(config))
    all_results.update(run_reward_benchmarks(config))
    
    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE (MLX)")
    print("=" * 60)
    print(f"{'Component':<20} {'Mean (ms)':>12} {'Std (ms)':>12}")
    print("-" * 60)
    for name, (mean, std) in all_results.items():
        print(f"{name:<20} {mean:>12.2f} {std:>12.2f}")
    
    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)
    
    return all_results


if __name__ == "__main__":
    run_all_benchmarks()
