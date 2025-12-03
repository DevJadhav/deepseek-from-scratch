"""
Modal Cloud GPU Integration for DeepSeek Training
==================================================

This package provides Modal cloud GPU integration for distributed training
with 10x A100-80GB GPUs using 5D parallelism (TP=2, PP=1, DP=5, EP=1, SP=1).

Usage
-----
Deploy and run training on Modal::

    # Deploy the Modal app
    uv run modal deploy modal/app.py
    
    # Run training with 10 GPUs
    uv run modal run modal/gpu_runner.py::run_distributed_training
    
    # Run PyTorch CUDA training
    uv run modal run modal/pytorch_cuda.py::train_pytorch
    
    # Run Rust CUDA training
    uv run modal run modal/rust_cuda.py::train_rust

Configuration
-------------
Set environment variables in `.env`::

    MODAL_TOKEN_ID=your-token-id
    MODAL_TOKEN_SECRET=your-token-secret

5D Parallelism Configuration
----------------------------
For 10 GPUs, the recommended split is:
- Tensor Parallel (TP) = 2: Split model weights within GPU pairs
- Pipeline Parallel (PP) = 1: No pipeline stages (small model)
- Data Parallel (DP) = 5: 5 data parallel replicas
- Expert Parallel (EP) = 1: Experts on same GPU (small MoE)
- Sequence Parallel (SP) = 1: No sequence splitting

Total GPUs: TP × PP × DP × EP = 2 × 1 × 5 × 1 = 10 GPUs
"""

# Only export config to avoid circular imports with Modal app
from modal_gpu.config import (
    ModalConfig,
    Parallelism5DConfig,
    get_modal_config,
    get_5d_config,
)

__all__ = [
    # Config
    "ModalConfig",
    "Parallelism5DConfig",
    "get_modal_config",
    "get_5d_config",
]
