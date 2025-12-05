"""
ANE Numerical Validation Module

Provides numerical comparison between ANE and PyTorch implementations:
- Tensor-level comparison with configurable tolerances
- Model-level validation across multiple inputs
- Statistical analysis of numerical differences
- Automated regression testing
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn


class ComparisonMetric(Enum):
    """Metrics for numerical comparison."""

    MAX_ABS_DIFF = "max_abs_diff"  # Maximum absolute difference
    MEAN_ABS_DIFF = "mean_abs_diff"  # Mean absolute difference
    RMSE = "rmse"  # Root mean squared error
    MAX_REL_DIFF = "max_rel_diff"  # Maximum relative difference
    MEAN_REL_DIFF = "mean_rel_diff"  # Mean relative difference
    COSINE_SIM = "cosine_sim"  # Cosine similarity
    CORRELATION = "correlation"  # Pearson correlation


@dataclass
class NumericalValidationConfig:
    """Configuration for numerical validation."""

    # Absolute tolerance
    atol: float = 1e-5

    # Relative tolerance
    rtol: float = 1e-4

    # Metrics to compute
    metrics: list[ComparisonMetric] = field(
        default_factory=lambda: [
            ComparisonMetric.MAX_ABS_DIFF,
            ComparisonMetric.MEAN_ABS_DIFF,
            ComparisonMetric.COSINE_SIM,
        ]
    )

    # Fail on first error vs collect all
    fail_fast: bool = False

    # Number of random inputs to test
    num_test_inputs: int = 10

    # Random seed for reproducibility
    seed: int = 42

    # Save detailed results
    save_results: bool = True
    results_dir: str = "validation_results"


@dataclass
class ValidationResult:
    """Result of numerical validation."""

    # Overall pass/fail
    passed: bool

    # Metrics computed
    metrics: dict[str, float]

    # Details per layer/output
    layer_results: dict[str, dict[str, float]] = field(default_factory=dict)

    # Test inputs used
    num_inputs_tested: int = 0

    # Number of failures
    num_failures: int = 0

    # Error messages
    errors: list[str] = field(default_factory=list)

    # Timing information
    reference_time_ms: float = 0.0
    target_time_ms: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "passed": self.passed,
            "metrics": self.metrics,
            "layer_results": self.layer_results,
            "num_inputs_tested": self.num_inputs_tested,
            "num_failures": self.num_failures,
            "errors": self.errors,
            "reference_time_ms": self.reference_time_ms,
            "target_time_ms": self.target_time_ms,
        }

    def save(self, path: str):
        """Save results to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def compare_tensors(
    reference: torch.Tensor,
    target: torch.Tensor,
    config: NumericalValidationConfig | None = None,
) -> ValidationResult:
    """
    Compare two tensors numerically.

    Args:
        reference: Reference tensor (ground truth)
        target: Target tensor to validate
        config: Validation configuration

    Returns:
        ValidationResult with comparison metrics
    """
    config = config or NumericalValidationConfig()

    # Ensure same device and dtype for comparison
    ref = reference.float().cpu()
    tgt = target.float().cpu()

    # Check shapes match
    if ref.shape != tgt.shape:
        return ValidationResult(
            passed=False,
            metrics={},
            errors=[f"Shape mismatch: reference {ref.shape} vs target {tgt.shape}"],
        )

    # Compute metrics
    metrics = {}
    diff = ref - tgt

    if ComparisonMetric.MAX_ABS_DIFF in config.metrics:
        metrics["max_abs_diff"] = diff.abs().max().item()

    if ComparisonMetric.MEAN_ABS_DIFF in config.metrics:
        metrics["mean_abs_diff"] = diff.abs().mean().item()

    if ComparisonMetric.RMSE in config.metrics:
        metrics["rmse"] = torch.sqrt((diff**2).mean()).item()

    if ComparisonMetric.MAX_REL_DIFF in config.metrics:
        # Avoid division by zero
        ref_abs = ref.abs()
        mask = ref_abs > 1e-10
        if mask.any():
            rel_diff = (diff.abs()[mask] / ref_abs[mask]).max().item()
        else:
            rel_diff = 0.0
        metrics["max_rel_diff"] = rel_diff

    if ComparisonMetric.MEAN_REL_DIFF in config.metrics:
        ref_abs = ref.abs()
        mask = ref_abs > 1e-10
        if mask.any():
            rel_diff = (diff.abs()[mask] / ref_abs[mask]).mean().item()
        else:
            rel_diff = 0.0
        metrics["mean_rel_diff"] = rel_diff

    if ComparisonMetric.COSINE_SIM in config.metrics:
        ref_flat = ref.flatten()
        tgt_flat = tgt.flatten()
        ref_norm = torch.norm(ref_flat)
        tgt_norm = torch.norm(tgt_flat)
        if ref_norm > 1e-10 and tgt_norm > 1e-10:
            cos_sim = torch.dot(ref_flat, tgt_flat) / (ref_norm * tgt_norm)
            metrics["cosine_sim"] = cos_sim.item()
        else:
            metrics["cosine_sim"] = 1.0 if torch.allclose(ref, tgt) else 0.0

    if ComparisonMetric.CORRELATION in config.metrics:
        ref_flat = ref.flatten()
        tgt_flat = tgt.flatten()
        ref_centered = ref_flat - ref_flat.mean()
        tgt_centered = tgt_flat - tgt_flat.mean()
        ref_std = ref_centered.std()
        tgt_std = tgt_centered.std()
        if ref_std > 1e-10 and tgt_std > 1e-10:
            corr = torch.dot(ref_centered, tgt_centered) / (
                len(ref_flat) * ref_std * tgt_std
            )
            metrics["correlation"] = corr.item()
        else:
            metrics["correlation"] = 1.0

    # Determine pass/fail
    passed = torch.allclose(ref, tgt, atol=config.atol, rtol=config.rtol)

    errors = []
    if not passed:
        errors.append(
            f"Tensors not close: max_diff={metrics.get('max_abs_diff', 'N/A')}, "
            f"atol={config.atol}, rtol={config.rtol}"
        )

    return ValidationResult(
        passed=passed,
        metrics=metrics,
        errors=errors,
    )


def compare_models(
    reference_model: nn.Module,
    target_model: nn.Module,
    input_generator: Callable[[], dict[str, torch.Tensor]],
    config: NumericalValidationConfig | None = None,
) -> ValidationResult:
    """
    Compare two models across multiple inputs.

    Args:
        reference_model: Reference model (ground truth)
        target_model: Target model to validate
        input_generator: Function that generates input dictionaries
        config: Validation configuration

    Returns:
        ValidationResult with aggregated metrics
    """
    config = config or NumericalValidationConfig()
    torch.manual_seed(config.seed)

    reference_model.eval()
    target_model.eval()

    all_metrics: dict[str, list[float]] = {}
    errors: list[str] = []
    num_failures = 0
    ref_times: list[float] = []
    tgt_times: list[float] = []

    with torch.no_grad():
        for i in range(config.num_test_inputs):
            inputs = input_generator()

            # Run reference
            start = time.perf_counter()
            ref_output = reference_model(**inputs)
            ref_times.append((time.perf_counter() - start) * 1000)

            # Run target
            start = time.perf_counter()
            tgt_output = target_model(**inputs)
            tgt_times.append((time.perf_counter() - start) * 1000)

            # Handle different output types
            if isinstance(ref_output, torch.Tensor):
                ref_tensors = {"output": ref_output}
                tgt_tensors = {"output": tgt_output}
            elif isinstance(ref_output, tuple):
                ref_tensors = {f"output_{j}": t for j, t in enumerate(ref_output)}
                tgt_tensors = {f"output_{j}": t for j, t in enumerate(tgt_output)}
            elif isinstance(ref_output, dict):
                ref_tensors = ref_output
                tgt_tensors = tgt_output
            else:
                errors.append(f"Unsupported output type: {type(ref_output)}")
                num_failures += 1
                continue

            # Compare each output tensor
            for key in ref_tensors:
                if key not in tgt_tensors:
                    errors.append(f"Input {i}: Missing key {key} in target output")
                    num_failures += 1
                    continue

                result = compare_tensors(ref_tensors[key], tgt_tensors[key], config)

                if not result.passed:
                    num_failures += 1
                    errors.extend([f"Input {i}, {key}: {e}" for e in result.errors])
                    if config.fail_fast:
                        return ValidationResult(
                            passed=False,
                            metrics={},
                            num_inputs_tested=i + 1,
                            num_failures=num_failures,
                            errors=errors,
                        )

                # Accumulate metrics
                for metric_name, value in result.metrics.items():
                    full_key = f"{key}_{metric_name}"
                    if full_key not in all_metrics:
                        all_metrics[full_key] = []
                    all_metrics[full_key].append(value)

    # Aggregate metrics
    aggregated_metrics = {}
    for key, values in all_metrics.items():
        aggregated_metrics[f"{key}_mean"] = sum(values) / len(values)
        aggregated_metrics[f"{key}_max"] = max(values)
        aggregated_metrics[f"{key}_min"] = min(values)

    passed = num_failures == 0

    return ValidationResult(
        passed=passed,
        metrics=aggregated_metrics,
        num_inputs_tested=config.num_test_inputs,
        num_failures=num_failures,
        errors=errors,
        reference_time_ms=sum(ref_times) / len(ref_times) if ref_times else 0,
        target_time_ms=sum(tgt_times) / len(tgt_times) if tgt_times else 0,
    )


class NumericalValidator:
    """
    Numerical validator for ANE vs PyTorch comparison.

    Example:
        validator = NumericalValidator(config)

        # Compare single tensors
        result = validator.compare_tensors(ref_tensor, ane_tensor)

        # Compare full models
        result = validator.compare_models(
            ref_model, ane_model,
            lambda: {"input_ids": torch.randint(0, 1000, (1, 128))}
        )

        # Save results
        validator.save_results("validation_results.json")
    """

    def __init__(self, config: NumericalValidationConfig | None = None):
        """Initialize validator with configuration."""
        self.config = config or NumericalValidationConfig()
        self.results: list[ValidationResult] = []

        if self.config.save_results:
            Path(self.config.results_dir).mkdir(parents=True, exist_ok=True)

    def compare_tensors(
        self,
        reference: torch.Tensor,
        target: torch.Tensor,
        name: str = "tensor",
    ) -> ValidationResult:
        """Compare two tensors and store result."""
        result = compare_tensors(reference, target, self.config)
        result.layer_results[name] = result.metrics
        self.results.append(result)
        return result

    def compare_models(
        self,
        reference_model: nn.Module,
        target_model: nn.Module,
        input_generator: Callable[[], dict[str, torch.Tensor]],
        name: str = "model",
    ) -> ValidationResult:
        """Compare two models and store result."""
        result = compare_models(
            reference_model, target_model, input_generator, self.config
        )
        result.layer_results[name] = result.metrics
        self.results.append(result)
        return result

    def compare_layer_outputs(
        self,
        reference_model: nn.Module,
        target_model: nn.Module,
        inputs: dict[str, torch.Tensor],
        layer_names: list[str] | None = None,
    ) -> ValidationResult:
        """
        Compare intermediate layer outputs between models.

        Args:
            reference_model: Reference model
            target_model: Target model
            inputs: Input tensors
            layer_names: Specific layers to compare (None = all)

        Returns:
            ValidationResult with per-layer metrics
        """
        reference_model.eval()
        target_model.eval()

        # Collect activations
        ref_activations: dict[str, torch.Tensor] = {}
        tgt_activations: dict[str, torch.Tensor] = {}

        def make_hook(storage: dict, name: str):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    storage[name] = output.detach()
                elif isinstance(output, tuple) and len(output) > 0:
                    storage[name] = output[0].detach()

            return hook

        # Register hooks
        ref_handles = []
        tgt_handles = []

        for name, module in reference_model.named_modules():
            if layer_names is None or name in layer_names:
                handle = module.register_forward_hook(make_hook(ref_activations, name))
                ref_handles.append(handle)

        for name, module in target_model.named_modules():
            if layer_names is None or name in layer_names:
                handle = module.register_forward_hook(make_hook(tgt_activations, name))
                tgt_handles.append(handle)

        # Forward pass
        with torch.no_grad():
            _ = reference_model(**inputs)
            _ = target_model(**inputs)

        # Clean up hooks
        for handle in ref_handles + tgt_handles:
            handle.remove()

        # Compare activations
        layer_results = {}
        errors = []
        num_failures = 0

        for name in ref_activations:
            if name not in tgt_activations:
                errors.append(f"Layer {name} missing in target model")
                num_failures += 1
                continue

            result = compare_tensors(
                ref_activations[name], tgt_activations[name], self.config
            )
            layer_results[name] = result.metrics

            if not result.passed:
                num_failures += 1
                errors.extend([f"Layer {name}: {e}" for e in result.errors])

        passed = num_failures == 0

        final_result = ValidationResult(
            passed=passed,
            metrics={},
            layer_results=layer_results,
            num_failures=num_failures,
            errors=errors,
        )

        self.results.append(final_result)
        return final_result

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all validation results."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)

        return {
            "total_validations": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0,
        }

    def save_results(self, filename: str | None = None):
        """Save all results to file."""
        if filename is None:
            filename = f"{self.config.results_dir}/validation_results.json"

        data = {
            "summary": self.get_summary(),
            "results": [r.to_dict() for r in self.results],
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

    def reset(self):
        """Clear stored results."""
        self.results.clear()
