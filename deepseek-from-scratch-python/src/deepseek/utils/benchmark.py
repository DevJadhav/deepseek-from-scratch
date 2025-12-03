"""
Comprehensive benchmark suite for DeepSeek from Scratch (Python/PyTorch)
Uses time.perf_counter() for precise timing with multiple warmup and measurement iterations.
"""

import torch
import time
import gc
from dataclasses import dataclass
from typing import Callable, Any

from deepseek.model.attention import MultiQueryAttention, GroupedQueryAttention
from deepseek.model.mla import MultiHeadLatentAttention, DeepSeekAttention
from deepseek.model.moe import DeepSeekMoE, StandardMoE
from deepseek.model.mtp import MTPModel
from deepseek.model.quantization import FP8Linear
from deepseek.training.grpo import GRPOTrainer
from deepseek.model.r1 import ReasoningModel
from deepseek.training.distillation import kd_loss_kl, kd_loss_mse, kd_loss_jsd


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarks."""
    batch_size: int = 4
    seq_len: int = 64
    d_model: int = 512
    warmup_iters: int = 5
    measure_iters: int = 20


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync_device(device: torch.device):
    """Synchronize device for accurate timing."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark_function(
    fn: Callable[[], Any],
    device: torch.device,
    warmup_iters: int = 5,
    measure_iters: int = 20,
) -> tuple[float, float]:
    """
    Benchmark a function with proper warmup and synchronization.
    Returns (mean_ms, std_ms).
    """
    # Warmup
    for _ in range(warmup_iters):
        _ = fn()
        sync_device(device)
    
    # Collect garbage before measuring
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    # Measure
    times = []
    for _ in range(measure_iters):
        sync_device(device)
        start = time.perf_counter()
        _ = fn()
        sync_device(device)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
    return mean_time, std_time


def run_attention_benchmarks(config: BenchmarkConfig, device: torch.device) -> dict:
    """Run benchmarks for all attention mechanisms."""
    results = {}
    x = torch.randn(config.batch_size, config.seq_len, config.d_model, device=device)
    
    print("\n" + "=" * 60)
    print("Attention Mechanism Benchmarks")
    print("=" * 60)
    print(f"Config: batch={config.batch_size}, seq_len={config.seq_len}, "
          f"d_model={config.d_model}")
    print(f"Warmup: {config.warmup_iters}, Measure: {config.measure_iters} iterations")
    print("-" * 60)
    
    # MQA
    mqa = MultiQueryAttention(d_model=config.d_model, num_heads=8).to(device)
    mean, std = benchmark_function(
        lambda: mqa(x), device, config.warmup_iters, config.measure_iters
    )
    results["MQA"] = (mean, std)
    print(f"MQA:          {mean:7.2f}ms ± {std:.2f}ms")
    
    # GQA
    gqa = GroupedQueryAttention(
        d_model=config.d_model, num_heads=32, num_groups=4
    ).to(device)
    mean, std = benchmark_function(
        lambda: gqa(x), device, config.warmup_iters, config.measure_iters
    )
    results["GQA"] = (mean, std)
    print(f"GQA:          {mean:7.2f}ms ± {std:.2f}ms")
    
    # MLA
    mla = MultiHeadLatentAttention(
        d_model=config.d_model, num_heads=8, d_latent=128, d_rope=64
    ).to(device)
    mean, std = benchmark_function(
        lambda: mla(x), device, config.warmup_iters, config.measure_iters
    )
    results["MLA"] = (mean, std)
    print(f"MLA:          {mean:7.2f}ms ± {std:.2f}ms")
    
    # DeepSeek Attention
    ds_attn = DeepSeekAttention(
        d_model=config.d_model, num_heads=8, d_latent=128, d_rope=64
    ).to(device)
    mean, std = benchmark_function(
        lambda: ds_attn(x), device, config.warmup_iters, config.measure_iters
    )
    results["DeepSeek Attn"] = (mean, std)
    print(f"DeepSeek Attn: {mean:6.2f}ms ± {std:.2f}ms")
    
    return results


def run_moe_benchmarks(config: BenchmarkConfig, device: torch.device) -> dict:
    """Run benchmarks for MoE variants."""
    results = {}
    x = torch.randn(config.batch_size, config.seq_len, config.d_model, device=device)
    
    print("\n" + "=" * 60)
    print("Mixture of Experts Benchmarks")
    print("=" * 60)
    print(f"Config: batch={config.batch_size}, seq_len={config.seq_len}, "
          f"d_model={config.d_model}")
    print("-" * 60)
    
    # Standard MoE
    std_moe = StandardMoE(
        d_model=config.d_model, d_hidden=config.d_model, 
        num_experts=22, top_k=2
    ).to(device)
    mean, std = benchmark_function(
        lambda: std_moe(x), device, config.warmup_iters, config.measure_iters
    )
    results["Standard MoE"] = (mean, std)
    print(f"Standard MoE:  {mean:7.2f}ms ± {std:.2f}ms")
    
    # DeepSeek MoE
    ds_moe = DeepSeekMoE(
        d_model=config.d_model, d_hidden=1024, 
        num_experts=10, num_shared=2, num_routed=8, top_k=2
    ).to(device)
    mean, std = benchmark_function(
        lambda: ds_moe(x), device, config.warmup_iters, config.measure_iters
    )
    results["DeepSeek MoE"] = (mean, std)
    print(f"DeepSeek MoE:  {mean:7.2f}ms ± {std:.2f}ms")
    
    return results


def run_mtp_benchmarks(config: BenchmarkConfig, device: torch.device) -> dict:
    """Run benchmarks for Multi-Token Prediction."""
    results = {}
    input_ids = torch.randint(0, 1000, (config.batch_size, config.seq_len), device=device)
    
    print("\n" + "=" * 60)
    print("Multi-Token Prediction Benchmarks")
    print("=" * 60)
    print(f"Config: batch={config.batch_size}, seq_len={config.seq_len}, "
          f"d_model={config.d_model}")
    print("-" * 60)
    
    # MTP with 1 future prediction
    mtp = MTPModel(
        vocab_size=1000, d_model=config.d_model, num_layers=2, k_predictions=1
    ).to(device)
    mean, std = benchmark_function(
        lambda: mtp(input_ids), device, config.warmup_iters, config.measure_iters
    )
    results["MTP (k=1)"] = (mean, std)
    print(f"MTP (k=1):     {mean:7.2f}ms ± {std:.2f}ms")
    
    # MTP with 2 future predictions
    mtp2 = MTPModel(
        vocab_size=1000, d_model=config.d_model, num_layers=2, k_predictions=2
    ).to(device)
    mean, std = benchmark_function(
        lambda: mtp2(input_ids), device, config.warmup_iters, config.measure_iters
    )
    results["MTP (k=2)"] = (mean, std)
    print(f"MTP (k=2):     {mean:7.2f}ms ± {std:.2f}ms")
    
    return results


def run_quantization_benchmarks(config: BenchmarkConfig, device: torch.device) -> dict:
    """Run benchmarks for FP8 quantization."""
    results = {}
    x = torch.randn(config.batch_size, config.seq_len, config.d_model, device=device)
    
    print("\n" + "=" * 60)
    print("FP8 Quantization Benchmarks")
    print("=" * 60)
    print(f"Config: batch={config.batch_size}, seq_len={config.seq_len}, "
          f"d_model={config.d_model}")
    print("-" * 60)
    
    # Regular Linear (baseline)
    linear = torch.nn.Linear(config.d_model, config.d_model).to(device)
    mean, std = benchmark_function(
        lambda: linear(x), device, config.warmup_iters, config.measure_iters
    )
    results["Linear (baseline)"] = (mean, std)
    print(f"Linear (FP32): {mean:7.2f}ms ± {std:.2f}ms")
    
    # FP8 Linear
    fp8 = FP8Linear(config.d_model, config.d_model).to(device)
    mean, std = benchmark_function(
        lambda: fp8(x), device, config.warmup_iters, config.measure_iters
    )
    results["FP8 Linear"] = (mean, std)
    print(f"FP8 Linear:    {mean:7.2f}ms ± {std:.2f}ms")
    
    return results


def run_grpo_benchmarks(config: BenchmarkConfig, device: torch.device) -> dict:
    """Run benchmarks for GRPO (Chapter 9)."""
    results = {}
    
    print("\n" + "=" * 60)
    print("GRPO (Group Relative Policy Optimization) Benchmarks")
    print("=" * 60)
    print(f"Config: batch={config.batch_size}, seq_len={config.seq_len}, "
          f"d_model={config.d_model}")
    print("-" * 60)
    
    vocab_size = 1000
    group_size = config.batch_size
    
    # Create GRPO trainer
    grpo = GRPOTrainer(beta=0.01)
    
    # Prepare inputs
    logits = torch.randn(group_size, config.seq_len, vocab_size, device=device)
    input_ids = torch.randint(0, vocab_size, (group_size, config.seq_len), device=device)
    rewards = torch.randn(group_size, device=device)
    ref_logits = torch.randn(group_size, config.seq_len, vocab_size, device=device)
    
    mean, std = benchmark_function(
        lambda: grpo.compute_loss(logits, input_ids, rewards, ref_logits),
        device, config.warmup_iters, config.measure_iters
    )
    results["GRPO Loss"] = (mean, std)
    print(f"GRPO Loss:     {mean:7.2f}ms ± {std:.2f}ms")
    
    return results


def run_r1_benchmarks(config: BenchmarkConfig, device: torch.device) -> dict:
    """Run benchmarks for R1 Reasoning Model (Chapter 9)."""
    results = {}
    
    print("\n" + "=" * 60)
    print("R1 Reasoning Model Benchmarks")
    print("=" * 60)
    print(f"Config: batch={config.batch_size}, seq_len={config.seq_len}, "
          f"d_model={config.d_model}")
    print("-" * 60)
    
    vocab_size = 1000
    
    # Create R1 model
    r1 = ReasoningModel(vocab_size=vocab_size, d_model=config.d_model).to(device)
    input_ids = torch.randint(0, vocab_size, (config.batch_size, config.seq_len), device=device)
    
    # R1 Forward pass
    mean, std = benchmark_function(
        lambda: r1(input_ids),
        device, config.warmup_iters, config.measure_iters
    )
    results["R1 Forward"] = (mean, std)
    print(f"R1 Forward:    {mean:7.2f}ms ± {std:.2f}ms")
    
    return results


def run_distillation_benchmarks(config: BenchmarkConfig, device: torch.device) -> dict:
    """Run benchmarks for Knowledge Distillation losses (Chapter 13)."""
    results = {}
    
    print("\n" + "=" * 60)
    print("Knowledge Distillation Benchmarks")
    print("=" * 60)
    print(f"Config: batch={config.batch_size}, seq_len={config.seq_len}, "
          f"d_model={config.d_model}")
    print("-" * 60)
    
    vocab_size = 1000
    
    # Prepare logits
    student_logits = torch.randn(config.batch_size, config.seq_len, vocab_size, device=device)
    teacher_logits = torch.randn(config.batch_size, config.seq_len, vocab_size, device=device)
    
    # KL Divergence Loss
    mean, std = benchmark_function(
        lambda: kd_loss_kl(student_logits, teacher_logits, temperature=4.0),
        device, config.warmup_iters, config.measure_iters
    )
    results["KD KL Loss"] = (mean, std)
    print(f"KD KL Loss:    {mean:7.2f}ms ± {std:.2f}ms")
    
    # MSE Loss
    mean, std = benchmark_function(
        lambda: kd_loss_mse(student_logits, teacher_logits, temperature=4.0),
        device, config.warmup_iters, config.measure_iters
    )
    results["KD MSE Loss"] = (mean, std)
    print(f"KD MSE Loss:   {mean:7.2f}ms ± {std:.2f}ms")
    
    # JSD Loss
    mean, std = benchmark_function(
        lambda: kd_loss_jsd(student_logits, teacher_logits, temperature=4.0),
        device, config.warmup_iters, config.measure_iters
    )
    results["KD JSD Loss"] = (mean, std)
    print(f"KD JSD Loss:   {mean:7.2f}ms ± {std:.2f}ms")
    
    return results


def run_dpo_benchmarks(config: BenchmarkConfig, device: torch.device) -> dict:
    """Run benchmarks for DPO loss computation (Chapter 12)."""
    results = {}
    
    print("\n" + "=" * 60)
    print("DPO (Direct Preference Optimization) Benchmarks")
    print("=" * 60)
    print(f"Config: batch={config.batch_size}, seq_len={config.seq_len}, "
          f"d_model={config.d_model}")
    print("-" * 60)
    
    # DPO loss computation (standalone function)
    beta = 0.1
    
    # Prepare log probabilities
    policy_chosen_logps = torch.randn(config.batch_size, device=device)
    policy_rejected_logps = torch.randn(config.batch_size, device=device)
    ref_chosen_logps = torch.randn(config.batch_size, device=device)
    ref_rejected_logps = torch.randn(config.batch_size, device=device)
    
    def compute_dpo_loss():
        # Log ratios
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps
        
        # DPO logits
        logits = beta * (chosen_logratios - rejected_logratios)
        
        # Standard DPO loss: -log(sigmoid(logits))
        losses = -torch.nn.functional.logsigmoid(logits)
        return losses.mean()
    
    mean, std = benchmark_function(
        compute_dpo_loss,
        device, config.warmup_iters, config.measure_iters
    )
    results["DPO Loss"] = (mean, std)
    print(f"DPO Loss:      {mean:7.2f}ms ± {std:.2f}ms")
    
    return results


def run_reward_benchmarks(config: BenchmarkConfig, device: torch.device) -> dict:
    """Run benchmarks for Reward Model (Chapter 12)."""
    results = {}
    
    print("\n" + "=" * 60)
    print("Reward Model Benchmarks")
    print("=" * 60)
    print(f"Config: batch={config.batch_size}, seq_len={config.seq_len}, "
          f"d_model={config.d_model}")
    print("-" * 60)
    
    # Create a simple reward head benchmark directly (simpler than full reward model)
    vocab_size = 1000
    
    # Simple reward head (from reward model architecture)
    reward_head = torch.nn.Sequential(
        torch.nn.Dropout(0.1),
        torch.nn.Linear(config.d_model, config.d_model // 4),
        torch.nn.GELU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(config.d_model // 4, 1),
    ).to(device)
    
    # Simulate hidden states
    hidden_states = torch.randn(config.batch_size, config.d_model, device=device)
    
    mean, std = benchmark_function(
        lambda: reward_head(hidden_states),
        device, config.warmup_iters, config.measure_iters
    )
    results["Reward Forward"] = (mean, std)
    print(f"Reward Forward: {mean:6.2f}ms ± {std:.2f}ms")
    
    return results


def run_all_benchmarks():
    """Run all benchmarks and print summary."""
    device = get_device()
    config = BenchmarkConfig()
    
    print("=" * 60)
    print("DeepSeek from Scratch - Python/PyTorch Benchmarks")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    
    all_results = {}
    
    # Run all benchmark suites
    all_results.update(run_attention_benchmarks(config, device))
    all_results.update(run_moe_benchmarks(config, device))
    all_results.update(run_mtp_benchmarks(config, device))
    all_results.update(run_quantization_benchmarks(config, device))
    all_results.update(run_grpo_benchmarks(config, device))
    all_results.update(run_r1_benchmarks(config, device))
    all_results.update(run_distillation_benchmarks(config, device))
    all_results.update(run_dpo_benchmarks(config, device))
    all_results.update(run_reward_benchmarks(config, device))
    
    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
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
