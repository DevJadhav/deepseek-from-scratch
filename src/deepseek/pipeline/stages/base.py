"""Abstract base class for pipeline stages."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

from deepseek.pipeline.config import Backend, PipelineConfig


@dataclass
class StageContext:
    """Shared context object passed between stages."""

    config: PipelineConfig
    previous_output: Optional[Any] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def get_framework_preset_for_backend(backend: Backend) -> str:
    """Get the appropriate framework preset based on the selected backend.
    
    This ensures no cross-backend fallback - if you select PyTorch,
    you stay in PyTorch (CUDA → MPS → CPU). If you select Rust,
    you stay in Rust (CUDA → Metal → CPU).
    
    Args:
        backend: The configured backend
        
    Returns:
        Framework preset name to use
    """
    # PyTorch backends -> PyTorch-only fallback
    if backend in (Backend.PYTORCH_CUDA, Backend.PYTORCH_MPS, Backend.PYTORCH_CPU):
        return "pytorch_only"
    
    # Rust backend -> Rust-only fallback  
    if backend == Backend.RUST:
        return "rust_only"
    
    # MLX -> Apple Silicon optimized (can mix MLX/PyTorch)
    if backend == Backend.MLX:
        return "apple_silicon"
    
    # Auto or Modal -> use heterogeneous (default)
    return "default"


class BaseStage(ABC):
    """Base class for all pipeline stages."""

    stage_name: str = "base"

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(f"ray_pipeline.stage.{self.stage_name}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def get_framework_preset(self) -> str:
        """Get the framework preset based on configured backend.
        
        Returns:
            Framework preset name that respects backend selection
        """
        return get_framework_preset_for_backend(self.config.backend)

    def setup(self):
        """Optional setup hook before running the stage."""

    @abstractmethod
    def run(self, context: StageContext) -> StageContext:
        """Execute the stage logic."""

    def teardown(self):
        """Optional cleanup hook after running the stage."""
