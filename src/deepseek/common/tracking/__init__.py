"""
DeepSeek Training Tracking Module
=================================

Comprehensive experiment tracking and profiling infrastructure.

Modules:
- wandb_tracker: Weights & Biases integration
- profiler: PyTorch profiler and NVTX integration
"""

from .wandb_tracker import (
    WandbConfig,
    WandbTracker,
    get_wandb_tracker,
    get_gpu_memory_stats,
    compute_mbu,
    create_sweep_config,
    run_sweep,
)

from .profiler import (
    ProfilerConfig,
    DeepSeekProfiler,
    MemoryTracker,
    LayerTimer,
    CommunicationProfiler,
    nvtx_range,
    nvtx_mark,
    nvtx_annotate,
    create_profiler,
    estimate_activation_memory,
)

__all__ = [
    # W&B Tracker
    "WandbConfig",
    "WandbTracker",
    "get_wandb_tracker",
    "get_gpu_memory_stats",
    "compute_mbu",
    # W&B Sweeps
    "create_sweep_config",
    "run_sweep",
    # Profiler
    "ProfilerConfig",
    "DeepSeekProfiler",
    "MemoryTracker",
    "LayerTimer",
    "CommunicationProfiler",
    "nvtx_range",
    "nvtx_mark",
    "nvtx_annotate",
    "create_profiler",
    "estimate_activation_memory",
]