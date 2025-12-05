"""
ANE Distribution Comparison Module

Provides output distribution comparison:
- KL divergence
- Jensen-Shannon divergence
- Histogram comparison
- Statistical tests
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DistributionConfig:
    """Configuration for distribution comparison."""

    # Number of histogram bins
    num_bins: int = 100

    # Temperature for softmax (for logit comparison)
    temperature: float = 1.0

    # Epsilon for numerical stability
    epsilon: float = 1e-10

    # Compare logits vs probabilities
    use_logits: bool = True

    # Number of samples for comparison
    num_samples: int = 1000


@dataclass
class DistributionResult:
    """Result of distribution comparison."""

    # KL divergence (P || Q)
    kl_divergence: float

    # Reverse KL (Q || P)
    kl_reverse: float

    # Jensen-Shannon divergence
    js_divergence: float

    # Total variation distance
    tv_distance: float

    # Histogram correlation
    histogram_correlation: float

    # Per-position statistics
    position_stats: dict[str, list[float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "kl_divergence": self.kl_divergence,
            "kl_reverse": self.kl_reverse,
            "js_divergence": self.js_divergence,
            "tv_distance": self.tv_distance,
            "histogram_correlation": self.histogram_correlation,
        }


def kl_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    epsilon: float = 1e-10,
) -> float:
    """
    Compute KL divergence D_KL(P || Q).

    Args:
        p: Reference distribution (probabilities)
        q: Target distribution (probabilities)
        epsilon: Small value for numerical stability

    Returns:
        KL divergence value
    """
    p = p.float().flatten()
    q = q.float().flatten()

    # Ensure valid probability distributions
    p = p.clamp(min=epsilon)
    q = q.clamp(min=epsilon)

    # Normalize
    p = p / p.sum()
    q = q / q.sum()

    # KL divergence
    kl = (p * torch.log(p / q)).sum()

    return kl.item()


def js_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    epsilon: float = 1e-10,
) -> float:
    """
    Compute Jensen-Shannon divergence.

    JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = 0.5 * (P + Q)

    Args:
        p: First distribution
        q: Second distribution
        epsilon: Small value for numerical stability

    Returns:
        JS divergence value (in range [0, ln(2)])
    """
    p = p.float().flatten()
    q = q.float().flatten()

    # Ensure valid probability distributions
    p = p.clamp(min=epsilon)
    q = q.clamp(min=epsilon)

    # Normalize
    p = p / p.sum()
    q = q / q.sum()

    # Mixture distribution
    m = 0.5 * (p + q)

    # JS divergence
    js = 0.5 * (p * torch.log(p / m)).sum() + 0.5 * (q * torch.log(q / m)).sum()

    return js.item()


def total_variation_distance(
    p: torch.Tensor,
    q: torch.Tensor,
) -> float:
    """
    Compute total variation distance.

    TV(P, Q) = 0.5 * sum(|P - Q|)

    Args:
        p: First distribution
        q: Second distribution

    Returns:
        TV distance (in range [0, 1])
    """
    p = p.float().flatten()
    q = q.float().flatten()

    # Normalize
    p = p / p.sum()
    q = q / q.sum()

    return 0.5 * (p - q).abs().sum().item()


def histogram_correlation(
    x: torch.Tensor,
    y: torch.Tensor,
    num_bins: int = 100,
) -> float:
    """
    Compute correlation between histograms of two distributions.

    Args:
        x: First set of values
        y: Second set of values
        num_bins: Number of histogram bins

    Returns:
        Pearson correlation coefficient
    """
    x = x.float().flatten()
    y = y.float().flatten()

    # Compute range
    min_val = min(x.min().item(), y.min().item())
    max_val = max(x.max().item(), y.max().item())

    # Compute histograms
    hist_x = torch.histc(x, bins=num_bins, min=min_val, max=max_val)
    hist_y = torch.histc(y, bins=num_bins, min=min_val, max=max_val)

    # Normalize
    hist_x = hist_x / hist_x.sum()
    hist_y = hist_y / hist_y.sum()

    # Correlation
    hx_centered = hist_x - hist_x.mean()
    hy_centered = hist_y - hist_y.mean()

    hx_std = hx_centered.std()
    hy_std = hy_centered.std()

    if hx_std > 1e-10 and hy_std > 1e-10:
        corr = torch.dot(hx_centered, hy_centered) / (num_bins * hx_std * hy_std)
        return corr.item()
    else:
        return 1.0 if torch.allclose(hist_x, hist_y) else 0.0


def compare_logit_distributions(
    ref_logits: torch.Tensor,
    tgt_logits: torch.Tensor,
    config: DistributionConfig | None = None,
) -> DistributionResult:
    """
    Compare logit distributions from two models.

    Args:
        ref_logits: Reference model logits [batch, seq, vocab]
        tgt_logits: Target model logits [batch, seq, vocab]
        config: Distribution comparison configuration

    Returns:
        DistributionResult with comparison metrics
    """
    config = config or DistributionConfig()

    # Convert to probabilities
    ref_probs = F.softmax(ref_logits / config.temperature, dim=-1)
    tgt_probs = F.softmax(tgt_logits / config.temperature, dim=-1)

    # Flatten to [num_positions, vocab_size]
    ref_probs = ref_probs.view(-1, ref_probs.shape[-1])
    tgt_probs = tgt_probs.view(-1, tgt_probs.shape[-1])

    # Sample positions for efficiency
    num_positions = ref_probs.shape[0]
    num_samples = min(config.num_samples, num_positions)

    if num_samples < num_positions:
        indices = torch.randperm(num_positions)[:num_samples]
        ref_probs = ref_probs[indices]
        tgt_probs = tgt_probs[indices]

    # Compute per-position metrics
    kl_values = []
    kl_reverse_values = []
    js_values = []
    tv_values = []

    for i in range(ref_probs.shape[0]):
        p = ref_probs[i]
        q = tgt_probs[i]

        kl_values.append(kl_divergence(p, q, config.epsilon))
        kl_reverse_values.append(kl_divergence(q, p, config.epsilon))
        js_values.append(js_divergence(p, q, config.epsilon))
        tv_values.append(total_variation_distance(p, q))

    # Aggregate metrics
    avg_kl = sum(kl_values) / len(kl_values) if kl_values else 0.0
    avg_kl_reverse = sum(kl_reverse_values) / len(kl_reverse_values) if kl_reverse_values else 0.0
    avg_js = sum(js_values) / len(js_values) if js_values else 0.0
    avg_tv = sum(tv_values) / len(tv_values) if tv_values else 0.0

    # Histogram correlation
    hist_corr = histogram_correlation(ref_logits, tgt_logits, config.num_bins)

    return DistributionResult(
        kl_divergence=avg_kl,
        kl_reverse=avg_kl_reverse,
        js_divergence=avg_js,
        tv_distance=avg_tv,
        histogram_correlation=hist_corr,
        position_stats={
            "kl_per_position": kl_values,
            "js_per_position": js_values,
        },
    )


class DistributionComparator:
    """
    Distribution comparator for model outputs.

    Example:
        comparator = DistributionComparator(config)

        # Compare logit distributions
        result = comparator.compare_logits(ref_logits, ane_logits)
        print(f"KL Divergence: {result.kl_divergence:.4f}")
        print(f"JS Divergence: {result.js_divergence:.4f}")

        # Compare model outputs
        result = comparator.compare_models(ref_model, ane_model, inputs)
    """

    def __init__(self, config: DistributionConfig | None = None):
        """Initialize comparator with configuration."""
        self.config = config or DistributionConfig()
        self.results: list[DistributionResult] = []

    def compare_logits(
        self,
        ref_logits: torch.Tensor,
        tgt_logits: torch.Tensor,
    ) -> DistributionResult:
        """Compare logit distributions."""
        result = compare_logit_distributions(ref_logits, tgt_logits, self.config)
        self.results.append(result)
        return result

    def compare_probabilities(
        self,
        ref_probs: torch.Tensor,
        tgt_probs: torch.Tensor,
    ) -> DistributionResult:
        """Compare probability distributions directly."""
        # Flatten
        ref_flat = ref_probs.view(-1, ref_probs.shape[-1])
        tgt_flat = tgt_probs.view(-1, tgt_probs.shape[-1])

        num_positions = ref_flat.shape[0]
        num_samples = min(self.config.num_samples, num_positions)

        if num_samples < num_positions:
            indices = torch.randperm(num_positions)[:num_samples]
            ref_flat = ref_flat[indices]
            tgt_flat = tgt_flat[indices]

        kl_values = []
        js_values = []
        tv_values = []

        for i in range(ref_flat.shape[0]):
            kl_values.append(kl_divergence(ref_flat[i], tgt_flat[i], self.config.epsilon))
            js_values.append(js_divergence(ref_flat[i], tgt_flat[i], self.config.epsilon))
            tv_values.append(total_variation_distance(ref_flat[i], tgt_flat[i]))

        result = DistributionResult(
            kl_divergence=sum(kl_values) / len(kl_values) if kl_values else 0.0,
            kl_reverse=0.0,  # Not computed for direct probs
            js_divergence=sum(js_values) / len(js_values) if js_values else 0.0,
            tv_distance=sum(tv_values) / len(tv_values) if tv_values else 0.0,
            histogram_correlation=histogram_correlation(ref_probs, tgt_probs, self.config.num_bins),
        )

        self.results.append(result)
        return result

    def compare_models(
        self,
        ref_model: nn.Module,
        tgt_model: nn.Module,
        inputs: dict[str, torch.Tensor],
    ) -> DistributionResult:
        """Compare output distributions from two models."""
        ref_model.eval()
        tgt_model.eval()

        with torch.no_grad():
            ref_output = ref_model(**inputs)
            tgt_output = tgt_model(**inputs)

        # Extract logits
        if isinstance(ref_output, torch.Tensor):
            ref_logits = ref_output
            tgt_logits = tgt_output
        elif hasattr(ref_output, "logits"):
            ref_logits = ref_output.logits
            tgt_logits = tgt_output.logits
        elif isinstance(ref_output, tuple):
            ref_logits = ref_output[0]
            tgt_logits = tgt_output[0]
        else:
            raise ValueError(f"Unsupported output type: {type(ref_output)}")

        return self.compare_logits(ref_logits, tgt_logits)

    def get_summary(self) -> dict:
        """Get summary of all comparisons."""
        if not self.results:
            return {}

        return {
            "num_comparisons": len(self.results),
            "avg_kl": sum(r.kl_divergence for r in self.results) / len(self.results),
            "avg_js": sum(r.js_divergence for r in self.results) / len(self.results),
            "avg_tv": sum(r.tv_distance for r in self.results) / len(self.results),
            "avg_hist_corr": sum(r.histogram_correlation for r in self.results) / len(self.results),
        }

    def reset(self):
        """Clear stored results."""
        self.results.clear()
