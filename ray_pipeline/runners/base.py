"""Base classes for pipeline runners."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ray_pipeline.config import PipelineConfig


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
