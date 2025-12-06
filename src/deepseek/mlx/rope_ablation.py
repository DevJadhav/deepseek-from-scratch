"""
RoPE Scaling Strategy Ablation Study Hooks (MLX)

This module provides hooks for comparing different RoPE scaling strategies on Apple Silicon.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum

import mlx.core as mx
import mlx.nn as nn


@dataclass
class AblationMetrics:
    """Metrics collected during ablation study"""
    strategy_name: str
    perplexity_by_length: dict[int, float] = field(default_factory=dict)
    attention_entropy: list[float] = field(default_factory=list)
    position_similarity_decay: list[float] = field(default_factory=list)
    memory_usage: int = 0
    forward_time_ms: float = 0.0
    context_utilization: list[float] = field(default_factory=list)


class RoPEStrategy(Enum):
    """RoPE scaling strategy variants"""
    STANDARD = "standard"
    LINEAR = "linear"
    NTK_AWARE = "ntk_aware"
    YARN = "yarn"
    DYNAMIC_NTK = "dynamic_ntk"


@dataclass
class LinearConfig:
    scale: float = 4.0


@dataclass
class NTKAwareConfig:
    alpha: float = 2.0


@dataclass
class YaRNConfig:
    scale: float = 4.0
    original_max_seq_len: int = 4096
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    attention_factor: float = 0.1


@dataclass
class DynamicNTKConfig:
    max_position_embeddings: int = 4096


@dataclass
class StrategyConfig:
    """Combined strategy configuration"""
    strategy: RoPEStrategy
    linear: LinearConfig | None = None
    ntk_aware: NTKAwareConfig | None = None
    yarn: YaRNConfig | None = None
    dynamic_ntk: DynamicNTKConfig | None = None

    @classmethod
    def standard(cls) -> StrategyConfig:
        return cls(strategy=RoPEStrategy.STANDARD)

    @classmethod
    def create_linear(cls, scale: float = 4.0) -> StrategyConfig:
        return cls(strategy=RoPEStrategy.LINEAR, linear=LinearConfig(scale=scale))

    @classmethod
    def create_ntk_aware(cls, alpha: float = 2.0) -> StrategyConfig:
        return cls(strategy=RoPEStrategy.NTK_AWARE, ntk_aware=NTKAwareConfig(alpha=alpha))

    @classmethod
    def create_yarn(cls, **kwargs) -> StrategyConfig:
        return cls(strategy=RoPEStrategy.YARN, yarn=YaRNConfig(**kwargs))

    @classmethod
    def create_dynamic_ntk(cls, max_position_embeddings: int = 4096) -> StrategyConfig:
        return cls(
            strategy=RoPEStrategy.DYNAMIC_NTK,
            dynamic_ntk=DynamicNTKConfig(max_position_embeddings=max_position_embeddings)
        )


@dataclass
class AblationConfig:
    """Configuration for ablation study"""
    eval_seq_lengths: list[int] = field(
        default_factory=lambda: [1024, 2048, 4096, 8192, 16384, 32768]
    )
    samples_per_length: int = 100
    measure_attention_entropy: bool = True
    measure_position_decay: bool = True
    verbose: bool = True
    strategies: list[StrategyConfig] = field(default_factory=lambda: [
        StrategyConfig.standard(),
        StrategyConfig.create_linear(scale=4.0),
        StrategyConfig.create_ntk_aware(alpha=2.0),
        StrategyConfig.create_yarn(),
        StrategyConfig.create_dynamic_ntk(max_position_embeddings=4096),
    ])


class RoPEAblationStudy:
    """Ablation study runner for RoPE scaling strategies on MLX"""

    def __init__(self, config: AblationConfig):
        self.config = config
        self.results: dict[str, AblationMetrics] = {}
        self.base = 10000.0

    def run(self) -> None:
        """Run ablation study"""
        for strategy_config in self.config.strategies:
            metrics = self._evaluate_strategy(strategy_config)
            self.results[strategy_config.strategy.value] = metrics

    def _evaluate_strategy(self, strategy_config: StrategyConfig) -> AblationMetrics:
        """Evaluate a single strategy"""
        metrics = AblationMetrics(strategy_name=strategy_config.strategy.value)
        d_head = 64

        total_time = 0.0
        for seq_len in self.config.eval_seq_lengths:
            if self.config.verbose:
                print(f"Evaluating {strategy_config.strategy.value} at seq_len={seq_len}")

            start = time.time()
            inv_freq = self._compute_inv_freq(strategy_config, d_head, seq_len)

            positions = mx.arange(seq_len, dtype=mx.float32)
            freqs = mx.outer(positions, inv_freq)
            cos = mx.cos(freqs)
            sin = mx.sin(freqs)

            mx.eval(cos, sin)  # Force evaluation
            total_time += (time.time() - start) * 1000

            if self.config.measure_attention_entropy:
                entropy = self._compute_position_entropy(cos, sin)
                metrics.attention_entropy.append(entropy)

            if self.config.measure_position_decay:
                decay = self._compute_position_decay(cos, sin, seq_len)
                metrics.position_similarity_decay.extend(decay)

            utilization = self._compute_context_utilization(cos, sin, seq_len)
            metrics.context_utilization.append(utilization)

        metrics.forward_time_ms = total_time / len(self.config.eval_seq_lengths)
        return metrics

    def _compute_inv_freq(
        self,
        strategy_config: StrategyConfig,
        d_head: int,
        max_seq_len: int,
    ) -> mx.array:
        """Compute inverse frequencies"""
        indices = mx.arange(0, d_head, 2, dtype=mx.float32)
        base_inv_freq = 1.0 / (self.base ** (indices / d_head))

        strategy = strategy_config.strategy

        if strategy == RoPEStrategy.STANDARD:
            return base_inv_freq

        elif strategy == RoPEStrategy.LINEAR:
            scale = strategy_config.linear.scale
            return base_inv_freq / scale

        elif strategy == RoPEStrategy.NTK_AWARE:
            alpha = strategy_config.ntk_aware.alpha
            new_base = self.base * (alpha ** (d_head / (d_head - 2)))
            return 1.0 / (new_base ** (indices / d_head))

        elif strategy == RoPEStrategy.YARN:
            cfg = strategy_config.yarn
            inv_freq_list = []

            for i in range(0, d_head, 2):
                dim_idx = i / d_head
                base_freq = 1.0 / (self.base ** dim_idx)
                wavelength = 2 * math.pi / base_freq

                low_freq_wavelen = cfg.original_max_seq_len / cfg.beta_slow
                high_freq_wavelen = cfg.original_max_seq_len / cfg.beta_fast

                if wavelength < high_freq_wavelen:
                    gamma = 0.0
                elif wavelength > low_freq_wavelen:
                    gamma = 1.0
                else:
                    gamma = (wavelength - high_freq_wavelen) / (low_freq_wavelen - high_freq_wavelen)

                scaled_freq = base_freq / cfg.scale
                final_freq = (1 - gamma) * base_freq + gamma * scaled_freq
                inv_freq_list.append(final_freq)

            return mx.array(inv_freq_list, dtype=mx.float32)

        elif strategy == RoPEStrategy.DYNAMIC_NTK:
            cfg = strategy_config.dynamic_ntk
            alpha = max(max_seq_len / cfg.max_position_embeddings, 1.0)
            new_base = self.base * (alpha ** (d_head / (d_head - 2)))
            return 1.0 / (new_base ** (indices / d_head))

        return base_inv_freq

    def _compute_position_entropy(self, cos: mx.array, sin: mx.array) -> float:
        """Compute entropy of position embeddings"""
        cos_var = float(mx.mean(mx.var(cos, axis=0)))
        sin_var = float(mx.mean(mx.var(sin, axis=0)))
        return (cos_var + sin_var) / 2.0

    def _compute_position_decay(
        self,
        cos: mx.array,
        sin: mx.array,
        seq_len: int,
    ) -> list[float]:
        """Compute position similarity decay"""
        decay = []
        sample_distances = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

        cos_0 = cos[0]
        sin_0 = sin[0]

        for dist in sample_distances:
            if dist >= seq_len:
                break
            cos_d = cos[dist]
            sin_d = sin[dist]

            cos_sim = float(mx.sum(cos_0 * cos_d))
            sin_sim = float(mx.sum(sin_0 * sin_d))

            similarity = (cos_sim + sin_sim) / 2.0
            decay.append(similarity)

        return decay

    def _compute_context_utilization(
        self,
        cos: mx.array,
        sin: mx.array,
        seq_len: int,
    ) -> float:
        """Compute context utilization metric"""
        if seq_len < 100:
            return 1.0

        positions = [0, seq_len // 4, seq_len // 2, 3 * seq_len // 4, seq_len - 1]
        total_distance = 0.0
        count = 0

        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                if positions[j] >= seq_len:
                    continue

                cos_i, cos_j = cos[positions[i]], cos[positions[j]]
                sin_i, sin_j = sin[positions[i]], sin[positions[j]]

                cos_diff = cos_i - cos_j
                sin_diff = sin_i - sin_j
                dist = float(mx.sqrt(mx.sum(cos_diff ** 2) + mx.sum(sin_diff ** 2)))

                total_distance += dist
                count += 1

        return total_distance / count if count > 0 else 0.0

    def get_results(self) -> dict[str, AblationMetrics]:
        return self.results

    def generate_report(self) -> str:
        """Generate summary report"""
        report = "# RoPE Scaling Strategy Ablation Study Report (MLX)\n\n"

        for name, metrics in self.results.items():
            report += f"## {name}\n\n"
            report += f"- Average forward time: {metrics.forward_time_ms:.2f} ms\n"

            if metrics.attention_entropy:
                avg_entropy = sum(metrics.attention_entropy) / len(metrics.attention_entropy)
                report += f"- Average attention entropy: {avg_entropy:.4f}\n"

            if metrics.context_utilization:
                avg_util = sum(metrics.context_utilization) / len(metrics.context_utilization)
                report += f"- Average context utilization: {avg_util:.4f}\n"

            report += "\n"

        return report


class AblationTrainingHook:
    """Hook for logging metrics during training"""

    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.metrics_log: list[tuple[int, str, float]] = []

    def log_metric(self, step: int, name: str, value: float) -> None:
        if step % self.log_interval == 0:
            self.metrics_log.append((step, name, value))

    def get_metrics(self) -> list[tuple[int, str, float]]:
        return self.metrics_log

    def export_csv(self) -> str:
        csv = "step,metric,value\n"
        for step, name, value in self.metrics_log:
            csv += f"{step},{name},{value}\n"
        return csv


class RoPEAblationModule(nn.Module):
    """Module wrapper for RoPE with ablation study support on MLX"""

    def __init__(
        self,
        d_head: int,
        max_seq_len: int,
        strategy_config: StrategyConfig | None = None,
        base: float = 10000.0,
    ):
        super().__init__()
        self.d_head = d_head
        self.max_seq_len = max_seq_len
        self.strategy_config = strategy_config or StrategyConfig.standard()
        self.base = base

        self.inv_freq = self._compute_inv_freq()
        self.mscale = self._compute_mscale()

    def _compute_inv_freq(self) -> mx.array:
        """Compute inverse frequencies"""
        indices = mx.arange(0, self.d_head, 2, dtype=mx.float32)
        strategy = self.strategy_config.strategy

        if strategy == RoPEStrategy.STANDARD:
            return 1.0 / (self.base ** (indices / self.d_head))

        elif strategy == RoPEStrategy.LINEAR:
            scale = self.strategy_config.linear.scale
            return 1.0 / (scale * self.base ** (indices / self.d_head))

        elif strategy == RoPEStrategy.NTK_AWARE:
            alpha = self.strategy_config.ntk_aware.alpha
            new_base = self.base * (alpha ** (self.d_head / (self.d_head - 2)))
            return 1.0 / (new_base ** (indices / self.d_head))

        elif strategy == RoPEStrategy.YARN:
            cfg = self.strategy_config.yarn
            inv_freq_list = []

            for i in range(0, self.d_head, 2):
                dim_idx = i / self.d_head
                base_freq = 1.0 / (self.base ** dim_idx)
                wavelength = 2 * math.pi / base_freq

                low_freq_wavelen = cfg.original_max_seq_len / cfg.beta_slow
                high_freq_wavelen = cfg.original_max_seq_len / cfg.beta_fast

                if wavelength < high_freq_wavelen:
                    gamma = 0.0
                elif wavelength > low_freq_wavelen:
                    gamma = 1.0
                else:
                    gamma = (wavelength - high_freq_wavelen) / (low_freq_wavelen - high_freq_wavelen)

                scaled_freq = base_freq / cfg.scale
                final_freq = (1 - gamma) * base_freq + gamma * scaled_freq
                inv_freq_list.append(final_freq)

            return mx.array(inv_freq_list, dtype=mx.float32)

        elif strategy == RoPEStrategy.DYNAMIC_NTK:
            cfg = self.strategy_config.dynamic_ntk
            alpha = max(self.max_seq_len / cfg.max_position_embeddings, 1.0)
            new_base = self.base * (alpha ** (self.d_head / (self.d_head - 2)))
            return 1.0 / (new_base ** (indices / self.d_head))

        return 1.0 / (self.base ** (indices / self.d_head))

    def _compute_mscale(self) -> float:
        """Compute attention scaling factor for YaRN"""
        if self.strategy_config.strategy == RoPEStrategy.YARN:
            cfg = self.strategy_config.yarn
            return cfg.attention_factor * math.log(cfg.scale) + 1.0
        return 1.0

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        """
        Apply RoPE to input tensor.

        Args:
            x: Input tensor [batch, heads, seq_len, d_head]
            offset: Position offset

        Returns:
            Position-encoded tensor
        """
        seq_len = x.shape[2]
        positions = mx.arange(offset, offset + seq_len, dtype=mx.float32)
        freqs = mx.outer(positions, self.inv_freq)

        cos = mx.cos(freqs) * self.mscale
        sin = mx.sin(freqs) * self.mscale

        # Reshape for broadcast
        cos = cos.reshape(1, 1, seq_len, self.d_head // 2)
        sin = sin.reshape(1, 1, seq_len, self.d_head // 2)

        # Split and rotate
        x_reshape = x.reshape(*x.shape[:-1], self.d_head // 2, 2)
        x_real = x_reshape[..., 0]
        x_imag = x_reshape[..., 1]

        out_real = x_real * cos - x_imag * sin
        out_imag = x_real * sin + x_imag * cos

        out = mx.stack([out_real, out_imag], axis=-1)
        return out.reshape(x.shape)

    def get_strategy_name(self) -> str:
        return self.strategy_config.strategy.value
