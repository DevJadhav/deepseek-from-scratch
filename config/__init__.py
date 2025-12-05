"""
Configuration presets for DeepSeek training.

Available configurations:
- tiny_mlx_config: ~10M parameter model with MLX backend
"""

from .tiny_mlx_config import (
    create_tiny_config,
    create_pretrain_only_config,
    create_quick_test_config,
)

__all__ = [
    "create_tiny_config",
    "create_pretrain_only_config", 
    "create_quick_test_config",
]
