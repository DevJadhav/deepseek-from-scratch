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
    ResourceRequirements,
    WaveConfig,
)
from .stages.base import StageContext
from .heterogeneous import (  # noqa: E402
    NodeArchitecture,
    DetectedResources,
    detect_resources,
    init_ray_with_resources,
    create_pipeline_parallel_placement_group,
    create_heterogeneous_placement_group,
    ClusterHealthMonitor,
)
from .data_ingestion import (  # noqa: E402
    ShuffleState,
    StreamingConfig,
    TokenBatcherConfig,
    DynamicPaddingConfig,
    DataIngestionConfig,
    DeterministicShuffler,
    StreamingDataPipeline,
    TokenLevelBatcher,
    DynamicPadder,
    DataIngestionPipeline,
)
from .training_loop import (  # noqa: E402
    # Expert Placement (Phase 2)
    HardwareTarget,
    ExpertLoadStats,
    ExpertLoadHistory,
    ExpertPlacementConfig,
    ExpertPlacementState,
    HeterogeneousExpertPlacement,
    # Checkpoint Interop (Phase 2)
    CheckpointFormat,
    CheckpointInteropConfig,
    CheckpointMetadata,
    TensorMetadata,
    CheckpointInterop,
    CANDLE_TO_PYTORCH_NAME_MAP,
    PYTORCH_TO_CANDLE_NAME_MAP,
)

__all__ = [
    "PipelineConfig",
    "ModelConfig", 
    "TrainingConfig",
    "DataConfig",
    "DistributedConfig",
    "ModelSize",
    "Backend",
    "ResourceRequirements",
    "WaveConfig",
    "StageContext",
    # Heterogeneous scheduling
    "NodeArchitecture",
    "DetectedResources",
    "detect_resources",
    "init_ray_with_resources",
    "create_pipeline_parallel_placement_group",
    "create_heterogeneous_placement_group",
    "ClusterHealthMonitor",
    # Data Ingestion (Phase 1)
    "ShuffleState",
    "StreamingConfig",
    "TokenBatcherConfig",
    "DynamicPaddingConfig",
    "DataIngestionConfig",
    "DeterministicShuffler",
    "StreamingDataPipeline",
    "TokenLevelBatcher",
    "DynamicPadder",
    "DataIngestionPipeline",
    # Expert Placement (Phase 2)
    "HardwareTarget",
    "ExpertLoadStats",
    "ExpertLoadHistory",
    "ExpertPlacementConfig",
    "ExpertPlacementState",
    "HeterogeneousExpertPlacement",
    # Checkpoint Interop (Phase 2)
    "CheckpointFormat",
    "CheckpointInteropConfig",
    "CheckpointMetadata",
    "TensorMetadata",
    "CheckpointInterop",
    "CANDLE_TO_PYTORCH_NAME_MAP",
    "PYTORCH_TO_CANDLE_NAME_MAP",
]
