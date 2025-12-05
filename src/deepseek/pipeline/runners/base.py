"""Base classes for pipeline runners."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from deepseek.pipeline.config import PipelineConfig


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Priority:
    1. DEEPSEEK_PROJECT_ROOT environment variable (for production/docker)
    2. Auto-detect by walking up from this file
    
    Returns:
        Path to project root directory
        
    Raises:
        RuntimeError: If project root cannot be determined
    """
    # Check environment variable first (production override)
    env_root = os.environ.get("DEEPSEEK_PROJECT_ROOT")
    if env_root:
        root = Path(env_root)
        if root.exists():
            return root
    
    # Auto-detect: walk up from this file to find pyproject.toml
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
        # Also check for rust-src as a marker
        if (parent / "rust-src").exists() and (parent / "src").exists():
            return parent
    
    # Fallback: assume 4 levels up from base.py (src/deepseek/pipeline/runners/base.py)
    fallback = Path(__file__).resolve().parents[4]
    if fallback.exists():
        return fallback
    
    raise RuntimeError(
        "Could not determine project root. Set DEEPSEEK_PROJECT_ROOT environment variable."
    )


@dataclass
class RunnerResult:
    """Standard result returned by backend runners."""

    metrics: Dict[str, float] = field(default_factory=dict)
    checkpoint_path: Optional[str] = None
    artifacts: Dict[str, str] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseRunner(ABC):
    """Abstract runner interface."""

    name: str = "base"

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.project_root = get_project_root()
        self.logger = logging.getLogger(f"ray_pipeline.runner.{self.name}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    @abstractmethod
    def run(self, **kwargs) -> RunnerResult:
        """Execute the backend-specific logic."""

    def setup(self):
        """Optional setup hook."""

    def teardown(self):
        """Optional teardown hook."""
