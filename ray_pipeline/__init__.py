"""Ray-based Production Pipeline for DeepSeek Training."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the Python implementation package is importable without installation.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_PY_SRC = _REPO_ROOT / "deepseek-from-scratch-python" / "src"
if _PY_SRC.exists() and str(_PY_SRC) not in sys.path:
    sys.path.insert(0, str(_PY_SRC))

__version__ = "0.1.0"

from .config import (  # noqa: E402  (import after path adjustment)
    PipelineConfig,
    ModelConfig,
    TrainingConfig,
    DataConfig,
    DistributedConfig,
    ModelSize,
    Backend,
)
from .stages.base import StageContext

__all__ = [
    "PipelineConfig",
    "ModelConfig", 
    "TrainingConfig",
    "DataConfig",
    "DistributedConfig",
    "ModelSize",
    "Backend",
    "StageContext",
]
