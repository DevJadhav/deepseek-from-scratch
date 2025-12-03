"""
Backend Runners

Each runner provides a backend-specific implementation for training:
- PyTorchRunner: For CUDA/MPS/CPU with PyTorch
- MLXRunner: For Apple Silicon with MLX
- RustRunner: For cross-platform with Rust/Candle
- ModalRunner: For cloud GPU training on Modal
"""

from .base import BaseRunner, RunnerResult
from .pytorch_runner import PyTorchRunner
from .mlx_runner import MLXRunner
from .rust_runner import RustRunner
from .modal_runner import ModalRunner

__all__ = [
    "BaseRunner",
    "RunnerResult",
    "PyTorchRunner",
    "MLXRunner",
    "RustRunner",
    "ModalRunner",
]
