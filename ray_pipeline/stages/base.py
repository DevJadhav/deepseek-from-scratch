"""Abstract base class for pipeline stages."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ray_pipeline.config import PipelineConfig


@dataclass
class StageContext:
    """Shared context object passed between stages."""

    config: PipelineConfig
    previous_output: Optional[Any] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


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

    def setup(self):
        """Optional setup hook before running the stage."""

    @abstractmethod
    def run(self, context: StageContext) -> StageContext:
        """Execute the stage logic."""

    def teardown(self):
        """Optional cleanup hook after running the stage."""
