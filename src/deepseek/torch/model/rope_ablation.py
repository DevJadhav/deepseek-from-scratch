"""
RoPE Scaling Strategy Ablation Study Hooks (PyTorch)

This module provides hooks for comparing different RoPE scaling strategies:
- Standard (no scaling)
- Linear interpolation
- NTK-aware scaling
- YaRN (Yet another RoPE extensioN)
- Dynamic NTK

Reference: DeepSeek-V3 architecture specification for 128K+ context support.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import math
import time
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


@dataclass
class AblationMetrics:
    """Metrics collected during ablation study"""
    # Name of the scaling strategy
    strategy_name: str
    # Perplexity at different sequence lengths
    perplexity_by_length: dict[int, float] = field(default_factory=dict)
    # Attention entropy (measure of attention spread)
    attention_entropy: list[float] = field(default_factory=list)
    # Position embedding similarity decay
    position_similarity_decay: list[float] = field(default_factory=list)
    # Memory usage (bytes)
    memory_usage: int = 0
    # Forward pass time (milliseconds)
    forward_time_ms: float = 0.0
    # Effective context utilization
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
    """Linear scaling configuration"""
    scale: float = 4.0


@dataclass
class NTKAwareConfig:
    """NTK-aware scaling configuration"""
    alpha: float = 2.0


@dataclass
class YaRNConfig:
    """YaRN scaling configuration"""
    scale: float = 4.0
    original_max_seq_len: int = 4096
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    attention_factor: float = 0.1


@dataclass
class DynamicNTKConfig:
    """Dynamic NTK scaling configuration"""
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
    def linear(cls, scale: float = 4.0) -> StrategyConfig:
        return cls(strategy=RoPEStrategy.LINEAR, linear=LinearConfig(scale=scale))
    
    @classmethod
    def ntk_aware(cls, alpha: float = 2.0) -> StrategyConfig:
        return cls(strategy=RoPEStrategy.NTK_AWARE, ntk_aware=NTKAwareConfig(alpha=alpha))
    
    @classmethod
    def yarn(cls, **kwargs) -> StrategyConfig:
        return cls(strategy=RoPEStrategy.YARN, yarn=YaRNConfig(**kwargs))
    
    @classmethod
    def dynamic_ntk(cls, max_position_embeddings: int = 4096) -> StrategyConfig:
        return cls(
            strategy=RoPEStrategy.DYNAMIC_NTK, 
            dynamic_ntk=DynamicNTKConfig(max_position_embeddings=max_position_embeddings)
        )


@dataclass
class AblationConfig:
    """Configuration for ablation study"""
    # Sequence lengths to evaluate
    eval_seq_lengths: list[int] = field(default_factory=lambda: [1024, 2048, 4096, 8192, 16384, 32768])
    # Number of samples per sequence length
    samples_per_length: int = 100
    # Whether to measure attention entropy
    measure_attention_entropy: bool = True
    # Whether to measure position similarity decay
    measure_position_decay: bool = True
    # Whether to log intermediate results
    verbose: bool = True
    # Strategies to compare
    strategies: list[StrategyConfig] = field(default_factory=lambda: [
        StrategyConfig.standard(),
        StrategyConfig.linear(scale=4.0),
        StrategyConfig.ntk_aware(alpha=2.0),
        StrategyConfig.yarn(),
        StrategyConfig.dynamic_ntk(max_position_embeddings=4096),
    ])


class RoPEAblationStudy:
    """Ablation study runner for RoPE scaling strategies"""
    
    def __init__(self, config: AblationConfig):
        self.config = config
        self.results: dict[str, AblationMetrics] = {}
        self.base = 10000.0
    
    def run(self, device: torch.device = torch.device("cpu")) -> None:
        """Run ablation study comparing all configured strategies"""
        for strategy_config in self.config.strategies:
            metrics = self._evaluate_strategy(strategy_config, device)
            self.results[strategy_config.strategy.value] = metrics
    
    def _evaluate_strategy(
        self, 
        strategy_config: StrategyConfig, 
        device: torch.device
    ) -> AblationMetrics:
        """Evaluate a single RoPE scaling strategy"""
        metrics = AblationMetrics(strategy_name=strategy_config.strategy.value)
        d_head = 64
        
        total_time = 0.0
        for seq_len in self.config.eval_seq_lengths:
            if self.config.verbose:
                print(f"Evaluating {strategy_config.strategy.value} at seq_len={seq_len}")
            
            # Compute RoPE embeddings
            start = time.time()
            inv_freq = self._compute_inv_freq(strategy_config, d_head, seq_len, device)
            
            # Generate positions
            positions = torch.arange(seq_len, device=device, dtype=torch.float32)
            
            # Compute frequencies
            freqs = torch.outer(positions, inv_freq)
            cos = torch.cos(freqs)
            sin = torch.sin(freqs)
            
            total_time += (time.time() - start) * 1000
            
            # Measure attention entropy
            if self.config.measure_attention_entropy:
                entropy = self._compute_position_entropy(cos, sin)
                metrics.attention_entropy.append(entropy)
            
            # Measure position decay
            if self.config.measure_position_decay:
                decay = self._compute_position_decay(cos, sin, seq_len)
                metrics.position_similarity_decay.extend(decay)
            
            # Compute context utilization
            utilization = self._compute_context_utilization(cos, sin, seq_len)
            metrics.context_utilization.append(utilization)
        
        metrics.forward_time_ms = total_time / len(self.config.eval_seq_lengths)
        return metrics
    
    def _compute_inv_freq(
        self,
        strategy_config: StrategyConfig,
        d_head: int,
        max_seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute inverse frequencies for a given strategy"""
        indices = torch.arange(0, d_head, 2, device=device, dtype=torch.float32)
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
            half_dim = d_head // 2
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
            
            return torch.tensor(inv_freq_list, device=device, dtype=torch.float32)
        
        elif strategy == RoPEStrategy.DYNAMIC_NTK:
            cfg = strategy_config.dynamic_ntk
            alpha = max(max_seq_len / cfg.max_position_embeddings, 1.0)
            new_base = self.base * (alpha ** (d_head / (d_head - 2)))
            return 1.0 / (new_base ** (indices / d_head))
        
        return base_inv_freq
    
    def _compute_position_entropy(self, cos: torch.Tensor, sin: torch.Tensor) -> float:
        """Compute entropy of position embeddings"""
        cos_var = cos.var(dim=0).mean().item()
        sin_var = sin.var(dim=0).mean().item()
        return (cos_var + sin_var) / 2.0
    
    def _compute_position_decay(
        self,
        cos: torch.Tensor,
        sin: torch.Tensor,
        seq_len: int,
    ) -> list[float]:
        """Compute position similarity decay curve"""
        decay = []
        sample_distances = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        
        cos_0 = cos[0]
        sin_0 = sin[0]
        
        for dist in sample_distances:
            if dist >= seq_len:
                break
            cos_d = cos[dist]
            sin_d = sin[dist]
            
            cos_sim = (cos_0 * cos_d).sum().item()
            sin_sim = (sin_0 * sin_d).sum().item()
            
            similarity = (cos_sim + sin_sim) / 2.0
            decay.append(similarity)
        
        return decay
    
    def _compute_context_utilization(
        self,
        cos: torch.Tensor,
        sin: torch.Tensor,
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
                dist = torch.sqrt((cos_diff ** 2).sum() + (sin_diff ** 2).sum()).item()
                
                total_distance += dist
                count += 1
        
        return total_distance / count if count > 0 else 0.0
    
    def get_results(self) -> dict[str, AblationMetrics]:
        """Get ablation study results"""
        return self.results
    
    def generate_report(self) -> str:
        """Generate summary report"""
        report = "# RoPE Scaling Strategy Ablation Study Report\n\n"
        
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
    """Hook for logging ablation study metrics during training"""
    
    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.metrics_log: list[tuple[int, str, float]] = []
    
    def log_metric(self, step: int, name: str, value: float) -> None:
        """Log a metric at the current training step"""
        if step % self.log_interval == 0:
            self.metrics_log.append((step, name, value))
    
    def get_metrics(self) -> list[tuple[int, str, float]]:
        """Get all logged metrics"""
        return self.metrics_log
    
    def export_csv(self) -> str:
        """Export metrics to CSV format"""
        csv = "step,metric,value\n"
        for step, name, value in self.metrics_log:
            csv += f"{step},{name},{value}\n"
        return csv


class RoPEAblationModule(nn.Module):
    """
    Module wrapper for RoPE with ablation study support.
    
    This module can be dropped into existing attention implementations
    to easily switch between RoPE scaling strategies for comparison.
    """
    
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
        
        # Compute and register inverse frequencies
        inv_freq = self._compute_inv_freq()
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute mscale for YaRN
        self.mscale = self._compute_mscale()
        
        # Cache for cos/sin
        self._cos_cache: torch.Tensor | None = None
        self._sin_cache: torch.Tensor | None = None
        self._cache_seq_len = 0
    
    def _compute_inv_freq(self) -> torch.Tensor:
        """Compute inverse frequencies based on strategy"""
        indices = torch.arange(0, self.d_head, 2, dtype=torch.float32)
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
            
            return torch.tensor(inv_freq_list, dtype=torch.float32)
        
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
    
    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        """Update cos/sin cache"""
        if seq_len <= self._cache_seq_len and self._cos_cache is not None:
            return
        
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        self._cos_cache = (emb.cos() * self.mscale).to(dtype)
        self._sin_cache = (emb.sin() * self.mscale).to(dtype)
        self._cache_seq_len = seq_len
    
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Apply RoPE to input tensor.
        
        Args:
            x: Input tensor [batch, heads, seq_len, d_head]
            offset: Position offset for KV cache
            
        Returns:
            Position-encoded tensor
        """
        seq_len = x.shape[2]
        self._update_cache(offset + seq_len, x.device, x.dtype)
        
        cos = self._cos_cache[offset:offset + seq_len].unsqueeze(0).unsqueeze(0)
        sin = self._sin_cache[offset:offset + seq_len].unsqueeze(0).unsqueeze(0)
        
        x1, x2 = x.chunk(2, dim=-1)
        rotated = torch.cat([-x2, x1], dim=-1)
        
        return x * cos + rotated * sin
    
    def get_strategy_name(self) -> str:
        """Get current strategy name"""
        return self.strategy_config.strategy.value
