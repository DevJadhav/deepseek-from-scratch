#!/usr/bin/env python3
"""
Ablation Study Infrastructure for DeepSeek-V3

This package provides a comprehensive ablation study framework for evaluating
architectural choices in the DeepSeek-V3 implementation.

Ablation Studies Available:
- Attention: MLA vs GQA vs MHA comparison
- Expert: 8 vs 64 vs 256 experts comparison
- Balancing: Auxiliary-loss-free vs auxiliary loss
- MTP: D=0, 1, 2, 3 comparison
- Precision: FP8 vs BF16 vs FP16

Usage:
    # Run all ablations
    uv run python scripts/ablation/run_all_ablations.py
    
    # Run specific ablation
    uv run python scripts/ablation/run_attention_ablation.py
    uv run python scripts/ablation/run_expert_ablation.py
    uv run python scripts/ablation/run_balancing_ablation.py
    uv run python scripts/ablation/run_mtp_ablation.py
    uv run python scripts/ablation/run_precision_ablation.py
"""

from .ablation_utils import (
    AblationConfig,
    AblationResult,
    AblationRunner,
    aggregate_results,
    generate_latex_table,
    plot_ablation_results,
    run_with_seeds,
    statistical_analysis,
)

__all__ = [
    "AblationConfig",
    "AblationResult",
    "AblationRunner",
    "aggregate_results",
    "generate_latex_table",
    "plot_ablation_results",
    "run_with_seeds",
    "statistical_analysis",
]
