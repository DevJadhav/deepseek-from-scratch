"""
Modal Cloud GPU Integration for DeepSeek Training
==================================================

This package provides Modal cloud GPU integration for distributed training
with A100-40GB GPUs using 5D parallelism and DualPipe bidirectional pipeline.

5D Parallelism Configuration
----------------------------
Initial (8 GPUs - $16.80/hr):
- Tensor Parallel (TP) = 2: Split model weights within GPU pairs
- Pipeline Parallel (PP) = 2: Enables DualPipe bidirectional scheduling
- Data Parallel (DP) = 2: 2 data parallel replicas
- Expert Parallel (EP) = 1: Experts on same GPU (small MoE)
- Sequence Parallel (SP) = 1: No sequence splitting

Total GPUs: TP × PP × DP × EP = 2 × 2 × 2 × 1 = 8 GPUs

Scaled (64 GPUs - $134.40/hr):
- TP=4, PP=4, DP=2, EP=2, SP=1
- Full DualPipe with MoE expert parallelism

Usage
-----
Deploy and run training on Modal::

    # Deploy the Modal app
    uv run modal deploy src/deepseek/cloud/modal/app.py
    
    # Run Ray cluster with PyTorch backend (8 GPUs)
    uv run modal run src/deepseek/cloud/modal/ray_cluster.py::deploy_ray_cluster --scale initial --backend pytorch
    
    # Run Ray cluster with Rust backend (8 GPUs)
    uv run modal run src/deepseek/cloud/modal/ray_cluster.py::deploy_ray_cluster --scale initial --backend rust
    
    # Scale up to 64 GPUs
    uv run modal run src/deepseek/cloud/modal/ray_cluster.py::deploy_ray_cluster --scale scaled --backend pytorch

Configuration
-------------
Set environment variables in `.env`::

    MODAL_TOKEN_ID=your-token-id
    MODAL_TOKEN_SECRET=your-token-secret
"""

# Configuration
from deepseek.cloud.modal.config import (
    ModalConfig,
    Parallelism5DConfig,
    get_modal_config,
    get_5d_config,
)

# Logging utilities
from deepseek.cloud.modal.logging_utils import (
    get_logger,
    configure_logging,
    TrainingLogger,
    get_correlation_id,
    set_correlation_id,
    generate_correlation_id,
)

# Training utilities
from deepseek.cloud.modal.training_utils import (
    build_model,
    load_training_data,
    build_deepspeed_config,
    get_model_size_config,
    get_default_training_config,
    ParquetDataset,
    collate_batch,
)

__all__ = [
    # Config
    "ModalConfig",
    "Parallelism5DConfig",
    "get_modal_config",
    "get_5d_config",
    # Logging
    "get_logger",
    "configure_logging",
    "TrainingLogger",
    "get_correlation_id",
    "set_correlation_id",
    "generate_correlation_id",
    # Training
    "build_model",
    "load_training_data",
    "build_deepspeed_config",
    "get_model_size_config",
    "get_default_training_config",
    "ParquetDataset",
    "collate_batch",
]

__all__ = [
    # Config
    "ModalConfig",
    "Parallelism5DConfig",
    "get_modal_config",
    "get_5d_config",
]
