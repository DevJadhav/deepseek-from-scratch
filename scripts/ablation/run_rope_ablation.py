#!/usr/bin/env python3
"""
RoPE Scaling Strategy Ablation Study

Compares different RoPE (Rotary Position Embedding) scaling strategies:
- Standard (no scaling) - baseline
- Linear interpolation - simple position scaling
- NTK-aware scaling - frequency-domain interpolation
- YaRN (Yet another RoPE extensioN) - attention factor + dynamic NTK
- Dynamic NTK - length-adaptive base frequency

Metrics:
- Perplexity at different context lengths
- Attention entropy (attention pattern spread)
- Position similarity decay (embedding coherence)
- Throughput (tokens/second)
- Memory usage

Usage:
    uv run python scripts/ablation/run_rope_ablation.py
    uv run python scripts/ablation/run_rope_ablation.py --seeds 42,123,456
    uv run python scripts/ablation/run_rope_ablation.py --max-steps 5000
    uv run python scripts/ablation/run_rope_ablation.py --backend mlx  # For Apple Silicon
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.ablation.ablation_utils import (
    AblationConfig,
    AblationRunner,
    aggregate_results,
    generate_latex_table,
    plot_ablation_results,
    statistical_analysis,
)


def create_rope_config(
    seeds: list[int],
    max_steps: int,
    output_dir: str,
    backend: str = "pytorch",
) -> AblationConfig:
    """Create RoPE ablation configuration."""

    # Base model configuration (small model for ablation)
    base_config = {
        "backend": backend,
        "model_size": "tiny",
        "vocab_size": 32000,
        "d_model": 256,
        "num_layers": 4,
        "num_heads": 8,
        "d_hidden": 1024,
        "use_moe": False,
        "batch_size": 8,
        "base_seq_len": 512,
        "rope_base": 10000.0,
        "rope_dim": 64,  # Per head
    }

    # RoPE scaling strategy variations
    variations = {
        "standard": {
            "rope_strategy": "standard",
            "description": "Standard RoPE (no scaling)",
        },
        "linear_2x": {
            "rope_strategy": "linear",
            "rope_scale": 2.0,
            "description": "Linear scaling 2x",
        },
        "linear_4x": {
            "rope_strategy": "linear",
            "rope_scale": 4.0,
            "description": "Linear scaling 4x",
        },
        "ntk_aware_a2": {
            "rope_strategy": "ntk_aware",
            "rope_alpha": 2.0,
            "description": "NTK-aware α=2",
        },
        "ntk_aware_a4": {
            "rope_strategy": "ntk_aware",
            "rope_alpha": 4.0,
            "description": "NTK-aware α=4",
        },
        "yarn_2x": {
            "rope_strategy": "yarn",
            "rope_scale": 2.0,
            "yarn_beta_fast": 32.0,
            "yarn_beta_slow": 1.0,
            "description": "YaRN 2x scaling",
        },
        "yarn_4x": {
            "rope_strategy": "yarn",
            "rope_scale": 4.0,
            "yarn_beta_fast": 32.0,
            "yarn_beta_slow": 1.0,
            "description": "YaRN 4x scaling",
        },
        "dynamic_ntk": {
            "rope_strategy": "dynamic_ntk",
            "rope_max_position_embeddings": 4096,
            "description": "Dynamic NTK",
        },
    }

    return AblationConfig(
        name="rope_scaling",
        description="Comparison of RoPE scaling strategies for extended context",
        base_config=base_config,
        variations=variations,
        seeds=seeds,
        max_steps=max_steps,
        eval_interval=100,
        output_dir=output_dir,
        log_to_wandb=True,
        wandb_project="deepseek-rope-ablation",
    )


def run_rope_ablation(config: AblationConfig) -> dict:
    """Run the RoPE ablation study."""
    
    # Create runner
    runner = AblationRunner(config)
    
    # Run all variations
    results = runner.run_all()
    
    return results


def analyze_rope_results(results: dict, config: AblationConfig) -> str:
    """Generate analysis report for RoPE ablation."""
    
    report = []
    report.append("=" * 60)
    report.append("RoPE Scaling Strategy Ablation Study Results")
    report.append("=" * 60)
    report.append("")
    
    # Aggregate results
    aggregated = aggregate_results(results)
    
    # Statistical analysis
    stats = statistical_analysis(aggregated)
    
    # Perplexity comparison
    report.append("Perplexity by Strategy:")
    report.append("-" * 40)
    for variation, metrics in aggregated.items():
        ppl_mean = metrics.get("perplexity_mean", 0)
        ppl_std = metrics.get("perplexity_std", 0)
        report.append(f"  {variation}: {ppl_mean:.2f} ± {ppl_std:.2f}")
    report.append("")
    
    # Context extension analysis
    report.append("Context Extension Performance:")
    report.append("-" * 40)
    for variation, metrics in aggregated.items():
        ctx_util = metrics.get("context_utilization_mean", 0)
        attn_entropy = metrics.get("attention_entropy_mean", 0)
        report.append(f"  {variation}:")
        report.append(f"    Context Utilization: {ctx_util:.2%}")
        report.append(f"    Attention Entropy: {attn_entropy:.3f}")
    report.append("")
    
    # Throughput comparison
    report.append("Throughput (tokens/sec):")
    report.append("-" * 40)
    for variation, metrics in aggregated.items():
        throughput = metrics.get("throughput_mean", 0)
        report.append(f"  {variation}: {throughput:.0f}")
    report.append("")
    
    # Best strategy recommendation
    best_by_ppl = min(aggregated.items(), 
                      key=lambda x: x[1].get("perplexity_mean", float("inf")))
    best_by_ctx = max(aggregated.items(),
                      key=lambda x: x[1].get("context_utilization_mean", 0))
    
    report.append("Recommendations:")
    report.append("-" * 40)
    report.append(f"  Best perplexity: {best_by_ppl[0]}")
    report.append(f"  Best context utilization: {best_by_ctx[0]}")
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(
        description="Run RoPE scaling strategy ablation study"
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,123,456",
        help="Comma-separated random seeds"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum training steps per variation"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./ablation_results/rope",
        help="Output directory for results"
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["pytorch", "mlx"],
        default="pytorch",
        help="Computation backend"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging"
    )
    args = parser.parse_args()
    
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    
    # Create configuration
    config = create_rope_config(
        seeds=seeds,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        backend=args.backend,
    )
    
    if args.no_wandb:
        config.log_to_wandb = False
    
    print(f"Running RoPE ablation with {len(config.variations)} variations")
    print(f"Seeds: {seeds}")
    print(f"Max steps: {args.max_steps}")
    print(f"Backend: {args.backend}")
    print()
    
    # Run ablation
    results = run_rope_ablation(config)
    
    # Analyze results
    report = analyze_rope_results(results, config)
    print(report)
    
    # Generate plots
    try:
        plot_ablation_results(
            results,
            output_path=f"{args.output_dir}/rope_ablation_plots.png"
        )
        print(f"\nPlots saved to {args.output_dir}/rope_ablation_plots.png")
    except Exception as e:
        print(f"Could not generate plots: {e}")
    
    # Generate LaTeX table
    try:
        latex = generate_latex_table(results, config)
        latex_path = f"{args.output_dir}/rope_ablation_table.tex"
        with open(latex_path, "w") as f:
            f.write(latex)
        print(f"LaTeX table saved to {latex_path}")
    except Exception as e:
        print(f"Could not generate LaTeX table: {e}")


if __name__ == "__main__":
    main()
