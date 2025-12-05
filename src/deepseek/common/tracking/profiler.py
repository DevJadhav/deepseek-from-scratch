"""
Profiling Infrastructure for DeepSeek Training
===============================================

Comprehensive profiling including:
- PyTorch Profiler with TensorBoard export
- NVIDIA Nsight Systems integration via NVTX markers
- Memory profiling with per-operation tracking
- Per-layer timing breakdown
- Communication profiling for distributed training
- Flame graph generation

Usage:
    from deepseek.common.tracking.profiler import DeepSeekProfiler, nvtx_range
    
    profiler = DeepSeekProfiler(config)
    
    with profiler:
        for step in range(max_steps):
            with nvtx_range("forward"):
                output = model(input)
            profiler.step()
"""

from __future__ import annotations

import contextlib
import functools
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import torch
import torch.nn as nn

# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class ProfilerConfig:
    """Configuration for profiling."""
    enabled: bool = True
    output_dir: str = "./profiler_output"
    
    # PyTorch Profiler settings
    pytorch_profiler: bool = True
    wait_steps: int = 1
    warmup_steps: int = 1
    active_steps: int = 3
    repeat: int = 1
    profile_memory: bool = True
    with_stack: bool = True
    record_shapes: bool = True
    with_flops: bool = True
    
    # TensorBoard export
    tensorboard_export: bool = True
    
    # NVTX settings
    nvtx_enabled: bool = True
    nvtx_colors: dict = field(default_factory=lambda: {
        "forward": 0x00FF00,   # Green
        "backward": 0xFF0000,  # Red
        "optimizer": 0x0000FF, # Blue
        "data": 0xFFFF00,      # Yellow
        "comm": 0xFF00FF,      # Magenta
    })
    
    # Memory profiling
    memory_profiling: bool = True
    memory_log_interval: int = 100
    reset_peak_on_log: bool = True
    
    # Per-layer profiling
    layer_profiling: bool = False
    
    # Continuous profiling
    continuous_mode: bool = False
    sample_every_n_steps: int = 1000


class NVTXRange:
    """
    Context manager for NVTX range markers.
    
    Used for profiling with NVIDIA Nsight Systems.
    """
    
    def __init__(self, name: str, color: int | None = None):
        self.name = name
        self.color = color
        self._active = False
        
    def __enter__(self):
        if torch.cuda.is_available():
            try:
                if self.color is not None:
                    torch.cuda.nvtx.range_push(self.name)
                else:
                    torch.cuda.nvtx.range_push(self.name)
                self._active = True
            except AttributeError:
                # NVTX not available
                pass
        return self
        
    def __exit__(self, *args):
        if self._active and torch.cuda.is_available():
            try:
                torch.cuda.nvtx.range_pop()
            except AttributeError:
                pass


def nvtx_range(name: str, color: int | None = None):
    """Create an NVTX range context manager."""
    return NVTXRange(name, color)


def nvtx_mark(name: str):
    """Create an NVTX mark (instant event)."""
    if torch.cuda.is_available():
        try:
            torch.cuda.nvtx.mark(name)
        except AttributeError:
            pass


def nvtx_annotate(name: str | None = None, color: int | None = None):
    """
    Decorator to add NVTX annotation to a function.
    
    Usage:
        @nvtx_annotate("my_function")
        def my_function():
            ...
    """
    def decorator(func: F) -> F:
        func_name = name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with nvtx_range(func_name, color):
                return func(*args, **kwargs)
        return wrapper  # type: ignore
    return decorator


class MemoryTracker:
    """
    Track GPU memory usage over training.
    
    Provides per-operation memory tracking and identifies
    peak memory layers.
    """
    
    def __init__(self, enabled: bool = True, log_interval: int = 100):
        self.enabled = enabled
        self.log_interval = log_interval
        self.step = 0
        self.memory_history: list[dict[str, float]] = []
        self.peak_memory: float = 0.0
        self.peak_step: int = 0
        
    def track(self) -> dict[str, float]:
        """Track current memory usage."""
        if not self.enabled or not torch.cuda.is_available():
            return {}
            
        stats = {
            "allocated_mb": torch.cuda.memory_allocated() / 1e6,
            "reserved_mb": torch.cuda.memory_reserved() / 1e6,
            "max_allocated_mb": torch.cuda.max_memory_allocated() / 1e6,
        }
        
        # Track peak
        if stats["max_allocated_mb"] > self.peak_memory:
            self.peak_memory = stats["max_allocated_mb"]
            self.peak_step = self.step
            
        self.memory_history.append({
            "step": self.step,
            **stats
        })
        
        self.step += 1
        return stats
        
    def reset_peak(self):
        """Reset peak memory tracking."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
    def get_summary(self) -> dict[str, Any]:
        """Get memory usage summary."""
        if not self.memory_history:
            return {}
            
        allocated = [h["allocated_mb"] for h in self.memory_history]
        return {
            "peak_memory_mb": self.peak_memory,
            "peak_step": self.peak_step,
            "mean_allocated_mb": sum(allocated) / len(allocated),
            "max_allocated_mb": max(allocated),
            "min_allocated_mb": min(allocated),
        }


class LayerTimer:
    """
    Time individual layers in a model.
    
    Useful for identifying bottlenecks and load imbalance.
    """
    
    def __init__(self, model: nn.Module, enabled: bool = True):
        self.model = model
        self.enabled = enabled
        self.layer_times: dict[str, list[float]] = {}
        self._hooks: list[Any] = []
        self._start_times: dict[str, float] = {}
        
    def enable(self):
        """Enable layer timing hooks."""
        if not self.enabled:
            return
            
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                self._register_hooks(name, module)
                
    def _register_hooks(self, name: str, module: nn.Module):
        """Register forward hooks for timing."""
        def pre_hook(mod, inp):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._start_times[name] = time.perf_counter()
            
        def post_hook(mod, inp, out):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - self._start_times.get(name, time.perf_counter())
            if name not in self.layer_times:
                self.layer_times[name] = []
            self.layer_times[name].append(elapsed * 1000)  # ms
            
        self._hooks.append(module.register_forward_pre_hook(pre_hook))
        self._hooks.append(module.register_forward_hook(post_hook))
        
    def disable(self):
        """Remove timing hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        
    def get_summary(self) -> dict[str, dict[str, float]]:
        """Get timing summary for all layers."""
        summary = {}
        for name, times in self.layer_times.items():
            if times:
                summary[name] = {
                    "mean_ms": sum(times) / len(times),
                    "max_ms": max(times),
                    "min_ms": min(times),
                    "total_ms": sum(times),
                    "calls": len(times),
                }
        return summary
        
    def get_top_k(self, k: int = 10) -> list[tuple[str, float]]:
        """Get top-k slowest layers by total time."""
        summary = self.get_summary()
        sorted_layers = sorted(
            summary.items(),
            key=lambda x: x[1]["total_ms"],
            reverse=True
        )
        return [(name, stats["total_ms"]) for name, stats in sorted_layers[:k]]


class DeepSeekProfiler:
    """
    Comprehensive profiler for DeepSeek training.
    
    Integrates:
    - PyTorch Profiler with TensorBoard export
    - NVTX markers for Nsight Systems
    - Memory tracking
    - Layer timing
    """
    
    def __init__(self, config: ProfilerConfig | dict[str, Any] | None = None):
        if config is None:
            config = ProfilerConfig()
        elif isinstance(config, dict):
            config = ProfilerConfig(**config)
        self.config = config
        
        self.profiler: torch.profiler.profile | None = None
        self.memory_tracker = MemoryTracker(
            enabled=config.memory_profiling,
            log_interval=config.memory_log_interval,
        )
        self.layer_timer: LayerTimer | None = None
        self._step_count = 0
        self._active = False
        
        # Create output directory
        if config.enabled:
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)
            
    def __enter__(self):
        """Start profiling."""
        if not self.config.enabled:
            return self
            
        if self.config.pytorch_profiler:
            # Build schedule
            schedule = torch.profiler.schedule(
                wait=self.config.wait_steps,
                warmup=self.config.warmup_steps,
                active=self.config.active_steps,
                repeat=self.config.repeat,
            )
            
            # Build trace handler
            if self.config.tensorboard_export:
                trace_handler = torch.profiler.tensorboard_trace_handler(
                    self.config.output_dir
                )
            else:
                def trace_handler(p):
                    output_path = Path(self.config.output_dir) / f"trace_{p.step_num}.json"
                    p.export_chrome_trace(str(output_path))
                    
            # Create profiler
            self.profiler = torch.profiler.profile(
                schedule=schedule,
                on_trace_ready=trace_handler,
                record_shapes=self.config.record_shapes,
                profile_memory=self.config.profile_memory,
                with_stack=self.config.with_stack,
                with_flops=self.config.with_flops,
            )
            self.profiler.__enter__()
            
        self._active = True
        return self
        
    def __exit__(self, *args):
        """Stop profiling."""
        if self.profiler is not None:
            self.profiler.__exit__(*args)
            self.profiler = None
            
        if self.layer_timer is not None:
            self.layer_timer.disable()
            
        self._active = False
        
    def step(self) -> dict[str, Any]:
        """
        Advance profiler step.
        
        Returns:
            Dictionary of profiling metrics for this step
        """
        metrics = {}
        
        if self.profiler is not None:
            self.profiler.step()
            
        # Track memory
        if self.config.memory_profiling and self._step_count % self.config.memory_log_interval == 0:
            metrics["memory"] = self.memory_tracker.track()
            if self.config.reset_peak_on_log:
                self.memory_tracker.reset_peak()
                
        self._step_count += 1
        return metrics
        
    def enable_layer_timing(self, model: nn.Module):
        """Enable per-layer timing for a model."""
        self.layer_timer = LayerTimer(model, enabled=self.config.layer_profiling)
        self.layer_timer.enable()
        
    def get_memory_summary(self) -> dict[str, Any]:
        """Get memory usage summary."""
        return self.memory_tracker.get_summary()
        
    def get_layer_summary(self) -> dict[str, dict[str, float]]:
        """Get layer timing summary."""
        if self.layer_timer is not None:
            return self.layer_timer.get_summary()
        return {}
        
    def get_slowest_layers(self, k: int = 10) -> list[tuple[str, float]]:
        """Get k slowest layers."""
        if self.layer_timer is not None:
            return self.layer_timer.get_top_k(k)
        return []
        
    @contextlib.contextmanager
    def profile_region(self, name: str):
        """
        Context manager for profiling a specific region.
        
        Adds both NVTX marker and records in PyTorch profiler.
        """
        with nvtx_range(name):
            if self.profiler is not None:
                with torch.profiler.record_function(name):
                    yield
            else:
                yield


class CommunicationProfiler:
    """
    Profile distributed communication patterns.
    
    Tracks all-reduce, all-gather, reduce-scatter, etc.
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.comm_times: dict[str, list[float]] = {
            "all_reduce": [],
            "all_gather": [],
            "reduce_scatter": [],
            "broadcast": [],
            "all_to_all": [],
        }
        
    @contextlib.contextmanager
    def track(self, operation: str):
        """Track a communication operation."""
        if not self.enabled:
            yield
            return
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            if operation in self.comm_times:
                self.comm_times[operation].append(elapsed * 1000)  # ms
                
    def get_summary(self) -> dict[str, dict[str, float]]:
        """Get communication timing summary."""
        summary = {}
        for op, times in self.comm_times.items():
            if times:
                summary[op] = {
                    "count": len(times),
                    "total_ms": sum(times),
                    "mean_ms": sum(times) / len(times),
                    "max_ms": max(times),
                }
        return summary


def create_profiler(config: dict[str, Any]) -> DeepSeekProfiler:
    """
    Create a profiler from configuration dictionary.
    
    Args:
        config: Configuration with 'profiling' key
        
    Returns:
        Configured DeepSeekProfiler
    """
    profiler_config = config.get("profiling", {})
    return DeepSeekProfiler(profiler_config)


# Activation memory estimation utilities
def estimate_activation_memory(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_layers: int,
    num_experts: int = 1,
    top_k: int = 1,
    dtype: torch.dtype = torch.bfloat16,
) -> float:
    """
    Estimate activation memory for a DeepSeek model.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        hidden_size: Model hidden size
        num_layers: Number of transformer layers
        num_experts: Number of MoE experts
        top_k: Experts selected per token
        dtype: Data type for activations
        
    Returns:
        Estimated activation memory in GB
    """
    bytes_per_element = 2 if dtype in [torch.float16, torch.bfloat16] else 4
    
    # Per-layer activation memory
    # Attention: Q, K, V, attention scores, output
    attention_mem = batch_size * seq_len * hidden_size * 5 * bytes_per_element
    
    # FFN/MoE: input, gate, up, down
    if num_experts > 1:
        # MoE: routing + expert computation
        ffn_mem = batch_size * seq_len * hidden_size * 4 * top_k * bytes_per_element
    else:
        ffn_mem = batch_size * seq_len * hidden_size * 4 * bytes_per_element
        
    # Norms, residuals
    other_mem = batch_size * seq_len * hidden_size * 2 * bytes_per_element
    
    per_layer_mem = attention_mem + ffn_mem + other_mem
    total_mem = per_layer_mem * num_layers
    
    return total_mem / 1e9  # GB
