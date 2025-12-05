#!/usr/bin/env python3
"""
Ablation Study Utilities

Common utilities for running ablation studies with proper statistical analysis,
result aggregation, and visualization.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class AblationConfig:
    """Configuration for an ablation study."""
    name: str
    description: str
    base_config: dict[str, Any]
    variations: dict[str, dict[str, Any]]  # name -> config overrides
    seeds: list[int] = field(default_factory=lambda: [42, 123, 456])
    max_steps: int = 1000
    eval_interval: int = 100
    output_dir: str = "./ablation_results"
    log_to_wandb: bool = True
    wandb_project: str = "deepseek-ablation"
    cache_completed: bool = True
    
    def get_cache_key(self, variation: str, seed: int) -> str:
        """Generate a unique cache key for a specific run."""
        config_str = json.dumps({
            "base": self.base_config,
            "variation": self.variations[variation],
            "seed": seed,
            "max_steps": self.max_steps,
        }, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]


@dataclass
class AblationResult:
    """Results from a single ablation run."""
    name: str
    variation: str
    seed: int
    metrics: dict[str, float]  # final metrics
    metric_history: dict[str, list[float]]  # step-by-step metrics
    config: dict[str, Any]
    runtime_seconds: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "variation": self.variation,
            "seed": self.seed,
            "metrics": self.metrics,
            "metric_history": self.metric_history,
            "config": self.config,
            "runtime_seconds": self.runtime_seconds,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AblationResult:
        return cls(**data)


class AblationRunner:
    """Runner for ablation studies with caching and logging."""
    
    def __init__(
        self,
        config: AblationConfig,
        train_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ):
        self.config = config
        self.train_fn = train_fn
        self.output_dir = Path(config.output_dir) / config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.output_dir / ".cache"
        self.cache_dir.mkdir(exist_ok=True)
        
    def _is_cached(self, variation: str, seed: int) -> bool:
        """Check if a run is cached."""
        if not self.config.cache_completed:
            return False
        cache_key = self.config.get_cache_key(variation, seed)
        cache_file = self.cache_dir / f"{cache_key}.json"
        return cache_file.exists()
    
    def _load_cached(self, variation: str, seed: int) -> AblationResult | None:
        """Load cached result."""
        cache_key = self.config.get_cache_key(variation, seed)
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return AblationResult.from_dict(json.load(f))
        return None
    
    def _save_cached(self, result: AblationResult) -> None:
        """Save result to cache."""
        cache_key = self.config.get_cache_key(result.variation, result.seed)
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
    
    def run_single(
        self,
        variation: str,
        seed: int,
        force: bool = False,
    ) -> AblationResult:
        """Run a single ablation configuration."""
        # Check cache
        if not force and self._is_cached(variation, seed):
            print(f"  Loading cached result for {variation} seed={seed}")
            cached = self._load_cached(variation, seed)
            if cached is not None:
                return cached
        
        print(f"  Running {variation} seed={seed}...")
        
        # Merge configs
        run_config = {**self.config.base_config}
        run_config.update(self.config.variations[variation])
        run_config["seed"] = seed
        run_config["max_steps"] = self.config.max_steps
        run_config["eval_interval"] = self.config.eval_interval
        
        # Initialize W&B if available
        if self.config.log_to_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.wandb_project,
                name=f"{self.config.name}_{variation}_seed{seed}",
                config=run_config,
                tags=[self.config.name, variation],
                reinit=True,
            )
        
        # Run training
        start_time = time.time()
        
        if self.train_fn is not None:
            results = self.train_fn(run_config)
        else:
            # Default: run CLI command
            results = self._run_cli(run_config)
        
        runtime = time.time() - start_time
        
        # Create result
        result = AblationResult(
            name=self.config.name,
            variation=variation,
            seed=seed,
            metrics=results.get("final_metrics", {}),
            metric_history=results.get("metric_history", {}),
            config=run_config,
            runtime_seconds=runtime,
        )
        
        # Log to W&B
        if self.config.log_to_wandb and WANDB_AVAILABLE:
            wandb.log({"final_metrics": result.metrics})
            wandb.finish()
        
        # Cache result
        self._save_cached(result)
        
        return result
    
    def _run_cli(self, config: dict[str, Any]) -> dict[str, Any]:
        """Run training via CLI command."""
        # Build command
        cmd = [
            "uv", "run", "python", "-m", "ray_pipeline.cli", "run",
            "--backend", config.get("backend", "pytorch"),
            "--max-steps", str(config.get("max_steps", 1000)),
            "--seed", str(config.get("seed", 42)),
        ]
        
        # Add config-specific arguments
        for key, value in config.items():
            if key not in ["backend", "max_steps", "seed", "eval_interval"]:
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        
        # Run command
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600 * 6,  # 6 hour timeout
            )
            
            # Parse output for metrics (simplified)
            return {
                "final_metrics": {"completed": result.returncode == 0},
                "metric_history": {},
            }
        except subprocess.TimeoutExpired:
            return {
                "final_metrics": {"completed": False, "timeout": True},
                "metric_history": {},
            }
    
    def run_all(self, force: bool = False) -> list[AblationResult]:
        """Run all variations with all seeds."""
        results = []
        
        total_runs = len(self.config.variations) * len(self.config.seeds)
        current_run = 0
        
        print(f"\n{'='*60}")
        print(f"ABLATION STUDY: {self.config.name}")
        print(f"{'='*60}")
        print(f"Description: {self.config.description}")
        print(f"Variations: {list(self.config.variations.keys())}")
        print(f"Seeds: {self.config.seeds}")
        print(f"Total runs: {total_runs}")
        print(f"{'='*60}\n")
        
        for variation in self.config.variations:
            print(f"\nVariation: {variation}")
            print("-" * 40)
            
            for seed in self.config.seeds:
                current_run += 1
                print(f"[{current_run}/{total_runs}]", end=" ")
                
                result = self.run_single(variation, seed, force=force)
                results.append(result)
        
        # Save all results
        self._save_results(results)
        
        return results
    
    def _save_results(self, results: list[AblationResult]) -> None:
        """Save all results to file."""
        output_file = self.output_dir / "results.json"
        with open(output_file, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        
        print(f"\nResults saved to: {output_file}")


def aggregate_results(results: list[AblationResult]) -> dict[str, dict[str, Any]]:
    """
    Aggregate results across seeds for each variation.
    
    Returns:
        Dictionary mapping variation name to aggregated metrics with mean, std, etc.
    """
    from collections import defaultdict
    
    # Group by variation
    by_variation = defaultdict(list)
    for result in results:
        by_variation[result.variation].append(result)
    
    aggregated = {}
    for variation, variation_results in by_variation.items():
        # Collect all metric values
        all_metrics = defaultdict(list)
        for result in variation_results:
            for key, value in result.metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    all_metrics[key].append(value)
        
        # Compute statistics
        variation_stats = {}
        for metric_name, values in all_metrics.items():
            values_array = np.array(values)
            variation_stats[metric_name] = {
                "mean": float(np.mean(values_array)),
                "std": float(np.std(values_array)),
                "min": float(np.min(values_array)),
                "max": float(np.max(values_array)),
                "n": len(values),
            }
        
        # Compute average runtime
        runtimes = [r.runtime_seconds for r in variation_results]
        variation_stats["runtime"] = {
            "mean": float(np.mean(runtimes)),
            "std": float(np.std(runtimes)),
            "n": len(runtimes),
        }
        
        aggregated[variation] = variation_stats
    
    return aggregated


def statistical_analysis(
    results: list[AblationResult],
    baseline_variation: str,
    test_variation: str,
    metric: str = "perplexity",
) -> dict[str, Any]:
    """
    Perform statistical significance testing between two variations.
    
    Uses independent samples t-test and reports:
    - t-statistic
    - p-value
    - Cohen's d effect size
    - 95% confidence interval
    """
    from scipy import stats
    
    # Get metric values for each variation
    baseline_values = [
        r.metrics.get(metric, np.nan)
        for r in results
        if r.variation == baseline_variation
    ]
    test_values = [
        r.metrics.get(metric, np.nan)
        for r in results
        if r.variation == test_variation
    ]
    
    # Remove NaN values
    baseline_values = [v for v in baseline_values if not np.isnan(v)]
    test_values = [v for v in test_values if not np.isnan(v)]
    
    if len(baseline_values) < 2 or len(test_values) < 2:
        return {
            "error": "Not enough samples for statistical analysis",
            "baseline_n": len(baseline_values),
            "test_n": len(test_values),
        }
    
    baseline_arr = np.array(baseline_values)
    test_arr = np.array(test_values)
    
    # t-test
    t_stat, p_value = stats.ttest_ind(baseline_arr, test_arr)
    
    # Cohen's d
    pooled_std = np.sqrt(
        ((len(baseline_arr) - 1) * np.var(baseline_arr) +
         (len(test_arr) - 1) * np.var(test_arr)) /
        (len(baseline_arr) + len(test_arr) - 2)
    )
    cohens_d = (np.mean(test_arr) - np.mean(baseline_arr)) / pooled_std if pooled_std > 0 else 0
    
    # Confidence interval for the difference
    diff_mean = np.mean(test_arr) - np.mean(baseline_arr)
    se_diff = np.sqrt(np.var(baseline_arr)/len(baseline_arr) + np.var(test_arr)/len(test_arr))
    ci_95 = (diff_mean - 1.96 * se_diff, diff_mean + 1.96 * se_diff)
    
    return {
        "baseline": {
            "mean": float(np.mean(baseline_arr)),
            "std": float(np.std(baseline_arr)),
            "n": len(baseline_arr),
        },
        "test": {
            "mean": float(np.mean(test_arr)),
            "std": float(np.std(test_arr)),
            "n": len(test_arr),
        },
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant_at_005": p_value < 0.05,
        "significant_at_001": p_value < 0.01,
        "cohens_d": float(cohens_d),
        "effect_size": (
            "large" if abs(cohens_d) > 0.8 else
            "medium" if abs(cohens_d) > 0.5 else
            "small"
        ),
        "difference_mean": float(diff_mean),
        "difference_ci_95": (float(ci_95[0]), float(ci_95[1])),
    }


def generate_latex_table(
    aggregated: dict[str, dict[str, Any]],
    metrics: list[str],
    caption: str = "Ablation Study Results",
    label: str = "tab:ablation",
) -> str:
    """
    Generate a LaTeX table from aggregated results.
    
    Args:
        aggregated: Output from aggregate_results()
        metrics: List of metric names to include
        caption: Table caption
        label: Table label for references
        
    Returns:
        LaTeX table string
    """
    # Build header
    header_cols = ["Variation"] + [m.replace("_", " ").title() for m in metrics]
    header = " & ".join(header_cols) + " \\\\"
    
    # Build rows
    rows = []
    for variation, stats in aggregated.items():
        row_data = [variation.replace("_", " ")]
        for metric in metrics:
            if metric in stats:
                mean = stats[metric]["mean"]
                std = stats[metric]["std"]
                row_data.append(f"${mean:.3f} \\pm {std:.3f}$")
            else:
                row_data.append("-")
        rows.append(" & ".join(row_data) + " \\\\")
    
    # Assemble table
    table = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{{'l' + 'c' * len(metrics)}}}
\\toprule
{header}
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    
    return table


def plot_ablation_results(
    aggregated: dict[str, dict[str, Any]],
    metric: str,
    output_path: str | None = None,
    title: str | None = None,
) -> None:
    """
    Create a bar plot of ablation results with error bars.
    
    Args:
        aggregated: Output from aggregate_results()
        metric: Metric to plot
        output_path: Path to save the figure
        title: Plot title
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping plot")
        return
    
    variations = list(aggregated.keys())
    means = []
    stds = []
    
    for variation in variations:
        if metric in aggregated[variation]:
            means.append(aggregated[variation][metric]["mean"])
            stds.append(aggregated[variation][metric]["std"])
        else:
            means.append(0)
            stds.append(0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(variations))
    
    bars = ax.bar(x, means, yerr=stds, capsize=5, color='steelblue', alpha=0.8)
    
    ax.set_xlabel('Variation', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(title or f'Ablation Study: {metric}', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace('_', '\n') for v in variations], fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.01 * max(means),
            f'{mean:.3f}',
            ha='center',
            va='bottom',
            fontsize=9,
        )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    plt.close()


def run_with_seeds(
    train_fn: Callable[[int], dict[str, Any]],
    seeds: list[int] | None = None,
    metric_key: str = "perplexity",
) -> dict[str, Any]:
    """
    Run training with multiple seeds and aggregate results.

    Args:
        train_fn: Training function that takes seed and returns metrics dict
        seeds: List of random seeds (default: [42, 123, 456])
        metric_key: Primary metric to aggregate

    Returns:
        Aggregated results with mean, std, and individual runs
    """
    if seeds is None:
        seeds = [42, 123, 456]
    results = []
    for seed in seeds:
        print(f"Running with seed {seed}...")
        metrics = train_fn(seed)
        results.append(metrics)
    
    # Aggregate primary metric
    values = [r.get(metric_key, np.nan) for r in results]
    values = [v for v in values if not np.isnan(v)]
    
    return {
        "mean": float(np.mean(values)) if values else np.nan,
        "std": float(np.std(values)) if values else np.nan,
        "min": float(np.min(values)) if values else np.nan,
        "max": float(np.max(values)) if values else np.nan,
        "n": len(values),
        "individual_runs": results,
    }
