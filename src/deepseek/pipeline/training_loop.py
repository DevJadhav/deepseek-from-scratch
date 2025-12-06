"""
Phase 2: Training (The MoE Loop) Implementation

This module implements the Phase 2 training components for production-ready
DeepSeek training:

1. HeterogeneousExpertPlacement: Intelligent expert placement across
   heterogeneous hardware (Apple Silicon + NVIDIA GPUs) based on load history

2. CheckpointInterop: Unified checkpoint format for Rust-PyTorch interoperability
   with automated conversion between Candle and PyTorch formats

3. ExpertLoadTracker: EMA-based expert load tracking for placement decisions

Reference: production_hardening.md Section 3.2 Phase 2: Training (The MoE Loop)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

# Optional imports with fallbacks
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

try:
    from safetensors import safe_open
    from safetensors.torch import save_file as safetensors_save

    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

LOGGER = logging.getLogger(__name__)


# =============================================================================
# Expert Placement Strategy
# =============================================================================


class HardwareTarget(Enum):
    """Target hardware for expert placement."""

    CUDA_H100 = "cuda_h100"  # High-end NVIDIA H100
    CUDA_A100 = "cuda_a100"  # NVIDIA A100
    CUDA_GENERIC = "cuda_generic"  # Generic CUDA GPU
    APPLE_SILICON = "apple_silicon"  # Apple Silicon (M1/M2/M3)
    CPU = "cpu"  # CPU fallback


@dataclass
class ExpertLoadStats:
    """Statistics for a single expert's load history."""

    expert_id: int
    total_tokens: int = 0
    total_activations: int = 0
    ema_load: float = 0.0
    peak_load: float = 0.0
    last_updated: float = 0.0

    def update(self, tokens_processed: int, ema_decay: float = 0.99) -> None:
        """Update load statistics with new observation."""
        current_load = float(tokens_processed)
        self.total_tokens += tokens_processed
        self.total_activations += 1

        # Update EMA
        if self.total_activations == 1:
            self.ema_load = current_load
        else:
            self.ema_load = ema_decay * self.ema_load + (1 - ema_decay) * current_load

        # Track peak
        self.peak_load = max(self.peak_load, current_load)
        self.last_updated = time.time()

    @property
    def average_load(self) -> float:
        """Get average load per activation."""
        if self.total_activations == 0:
            return 0.0
        return self.total_tokens / self.total_activations


@dataclass
class ExpertLoadHistory:
    """Tracks load history for all experts over time."""

    num_experts: int
    ema_decay: float = 0.99
    stats: dict[int, ExpertLoadStats] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize stats for all experts."""
        for i in range(self.num_experts):
            self.stats[i] = ExpertLoadStats(expert_id=i)

    def record_batch(self, expert_token_counts: dict[int, int]) -> None:
        """Record token counts for a batch.

        Args:
            expert_token_counts: Mapping of expert_id -> tokens processed
        """
        for expert_id, count in expert_token_counts.items():
            if expert_id in self.stats:
                self.stats[expert_id].update(count, self.ema_decay)

    def get_load_ranking(self) -> list[tuple[int, float]]:
        """Get experts ranked by EMA load (descending).

        Returns:
            List of (expert_id, ema_load) tuples sorted by load
        """
        return sorted(
            [(eid, stats.ema_load) for eid, stats in self.stats.items()],
            key=lambda x: x[1],
            reverse=True,
        )

    def get_hot_cold_split(
        self,
        hot_fraction: float = 0.2,
    ) -> tuple[list[int], list[int]]:
        """Split experts into hot (high load) and cold (low load) groups.

        Args:
            hot_fraction: Fraction of experts to consider "hot" (default 20%)

        Returns:
            Tuple of (hot_expert_ids, cold_expert_ids)
        """
        ranking = self.get_load_ranking()
        cutoff = max(1, int(len(ranking) * hot_fraction))

        hot_experts = [eid for eid, _ in ranking[:cutoff]]
        cold_experts = [eid for eid, _ in ranking[cutoff:]]

        return hot_experts, cold_experts

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "num_experts": self.num_experts,
            "ema_decay": self.ema_decay,
            "stats": {
                str(eid): {
                    "expert_id": stats.expert_id,
                    "total_tokens": stats.total_tokens,
                    "total_activations": stats.total_activations,
                    "ema_load": stats.ema_load,
                    "peak_load": stats.peak_load,
                    "last_updated": stats.last_updated,
                }
                for eid, stats in self.stats.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExpertLoadHistory:
        """Deserialize from dictionary."""
        history = cls(
            num_experts=data["num_experts"],
            ema_decay=data.get("ema_decay", 0.99),
        )
        for eid_str, stats_dict in data.get("stats", {}).items():
            eid = int(eid_str)
            stats = ExpertLoadStats(
                expert_id=stats_dict["expert_id"],
                total_tokens=stats_dict["total_tokens"],
                total_activations=stats_dict["total_activations"],
                ema_load=stats_dict["ema_load"],
                peak_load=stats_dict["peak_load"],
                last_updated=stats_dict["last_updated"],
            )
            history.stats[eid] = stats
        return history


@dataclass
class ExpertPlacementConfig:
    """Configuration for heterogeneous expert placement."""

    # Hot/cold split parameters
    hot_fraction: float = 0.2  # Top 20% are "hot"

    # Hardware assignment
    hot_expert_target: HardwareTarget = HardwareTarget.CUDA_H100
    cold_expert_target: HardwareTarget = HardwareTarget.APPLE_SILICON
    shared_expert_target: HardwareTarget = HardwareTarget.CUDA_H100

    # Load tracking
    ema_decay: float = 0.99
    min_activations_for_placement: int = 100  # Minimum samples before placement decisions

    # Rebalancing
    rebalance_interval_steps: int = 1000
    rebalance_threshold: float = 0.1  # Min load change to trigger rebalance

    # Expert consolidation for memory optimization
    enable_consolidation: bool = False
    consolidation_factor: int = 4


@dataclass
class ExpertPlacementState:
    """Current state of expert placement."""

    expert_to_hardware: dict[int, HardwareTarget] = field(default_factory=dict)
    shared_expert_hardware: HardwareTarget = HardwareTarget.CUDA_H100
    last_rebalance_step: int = 0
    placement_version: int = 0

    def get_experts_for_hardware(self, target: HardwareTarget) -> list[int]:
        """Get list of expert IDs assigned to a hardware target."""
        return [eid for eid, hw in self.expert_to_hardware.items() if hw == target]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "expert_to_hardware": {str(k): v.value for k, v in self.expert_to_hardware.items()},
            "shared_expert_hardware": self.shared_expert_hardware.value,
            "last_rebalance_step": self.last_rebalance_step,
            "placement_version": self.placement_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExpertPlacementState:
        """Deserialize from dictionary."""
        return cls(
            expert_to_hardware={
                int(k): HardwareTarget(v) for k, v in data.get("expert_to_hardware", {}).items()
            },
            shared_expert_hardware=HardwareTarget(data.get("shared_expert_hardware", "cuda_h100")),
            last_rebalance_step=data.get("last_rebalance_step", 0),
            placement_version=data.get("placement_version", 0),
        )


class HeterogeneousExpertPlacement:
    """Manages expert placement across heterogeneous hardware.

    This class implements the expert placement strategy from production_hardening.md:
    - Shared experts (always active) → H100 CUDA (high compute)
    - Hot routed experts (top 20%) → H100 CUDA
    - Cold routed experts (bottom 80%) → Apple Silicon (cost-efficient)

    The placement is dynamically updated based on observed load patterns.

    Example:
        ```python
        # Initialize
        placement = HeterogeneousExpertPlacement(
            num_experts=256,
            num_shared_experts=2,
        )

        # During training, record expert loads
        placement.record_expert_loads({0: 100, 1: 50, 2: 200, ...})

        # Get current placement
        cuda_experts = placement.get_experts_for_hardware(HardwareTarget.CUDA_H100)
        metal_experts = placement.get_experts_for_hardware(HardwareTarget.APPLE_SILICON)

        # Check if rebalancing needed
        if placement.should_rebalance(current_step=5000):
            placement.rebalance()
        ```
    """

    def __init__(
        self,
        num_experts: int,
        num_shared_experts: int = 1,
        config: ExpertPlacementConfig | None = None,
    ):
        """Initialize expert placement manager.

        Args:
            num_experts: Number of routed experts
            num_shared_experts: Number of shared experts (always active)
            config: Placement configuration
        """
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.config = config or ExpertPlacementConfig()

        # Initialize load history
        self.load_history = ExpertLoadHistory(
            num_experts=num_experts,
            ema_decay=self.config.ema_decay,
        )

        # Initialize placement state with default assignment
        self.state = ExpertPlacementState(
            shared_expert_hardware=self.config.shared_expert_target,
        )

        # Initial placement: all to hot target until we have data
        for i in range(num_experts):
            self.state.expert_to_hardware[i] = self.config.hot_expert_target

        LOGGER.info(
            "Initialized HeterogeneousExpertPlacement: %d routed + %d shared experts",
            num_experts,
            num_shared_experts,
        )

    def record_expert_loads(
        self,
        expert_token_counts: dict[int, int],
        step: int | None = None,
    ) -> None:
        """Record expert load statistics from a batch.

        Args:
            expert_token_counts: Mapping of expert_id -> tokens processed
            step: Optional current training step for rebalance checks
        """
        self.load_history.record_batch(expert_token_counts)

        # Auto-rebalance if step provided
        if step is not None and self.should_rebalance(step):
            self.rebalance(step)

    def should_rebalance(self, current_step: int) -> bool:
        """Check if expert placement should be rebalanced.

        Args:
            current_step: Current training step

        Returns:
            True if rebalancing is recommended
        """
        # Check minimum samples
        total_activations = sum(s.total_activations for s in self.load_history.stats.values())
        if total_activations < self.config.min_activations_for_placement:
            return False

        # Check interval
        steps_since_rebalance = current_step - self.state.last_rebalance_step
        return steps_since_rebalance >= self.config.rebalance_interval_steps

    def rebalance(self, current_step: int | None = None) -> dict[str, Any]:
        """Rebalance expert placement based on load history.

        Args:
            current_step: Current training step (for tracking)

        Returns:
            Dictionary with rebalance results including moved experts
        """
        # Get hot/cold split
        hot_experts, cold_experts = self.load_history.get_hot_cold_split(
            hot_fraction=self.config.hot_fraction
        )

        # Track changes
        moved_experts: list[dict[str, Any]] = []

        # Update placement
        old_placement = dict(self.state.expert_to_hardware)

        for eid in hot_experts:
            new_hw = self.config.hot_expert_target
            if old_placement.get(eid) != new_hw:
                moved_experts.append(
                    {
                        "expert_id": eid,
                        "from": old_placement.get(eid, HardwareTarget.CUDA_H100).value,
                        "to": new_hw.value,
                        "ema_load": self.load_history.stats[eid].ema_load,
                    }
                )
            self.state.expert_to_hardware[eid] = new_hw

        for eid in cold_experts:
            new_hw = self.config.cold_expert_target
            if old_placement.get(eid) != new_hw:
                moved_experts.append(
                    {
                        "expert_id": eid,
                        "from": old_placement.get(eid, HardwareTarget.CUDA_H100).value,
                        "to": new_hw.value,
                        "ema_load": self.load_history.stats[eid].ema_load,
                    }
                )
            self.state.expert_to_hardware[eid] = new_hw

        # Update state
        self.state.placement_version += 1
        if current_step is not None:
            self.state.last_rebalance_step = current_step

        result = {
            "placement_version": self.state.placement_version,
            "hot_experts": hot_experts,
            "cold_experts": cold_experts,
            "moved_experts": moved_experts,
            "cuda_count": len(hot_experts),
            "metal_count": len(cold_experts),
        }

        LOGGER.info(
            "Rebalanced experts: %d hot (CUDA), %d cold (Metal), %d moved",
            len(hot_experts),
            len(cold_experts),
            len(moved_experts),
        )

        return result

    def get_experts_for_hardware(self, target: HardwareTarget) -> list[int]:
        """Get expert IDs assigned to a specific hardware target.

        Args:
            target: Hardware target to query

        Returns:
            List of expert IDs
        """
        return self.state.get_experts_for_hardware(target)

    def get_cuda_experts(self) -> list[int]:
        """Get expert IDs assigned to any CUDA hardware."""
        cuda_targets = {
            HardwareTarget.CUDA_H100,
            HardwareTarget.CUDA_A100,
            HardwareTarget.CUDA_GENERIC,
        }
        return [eid for eid, hw in self.state.expert_to_hardware.items() if hw in cuda_targets]

    def get_metal_experts(self) -> list[int]:
        """Get expert IDs assigned to Apple Silicon."""
        return self.get_experts_for_hardware(HardwareTarget.APPLE_SILICON)

    def get_placement_summary(self) -> dict[str, Any]:
        """Get summary of current placement."""
        by_hardware: dict[str, list[int]] = defaultdict(list)
        for eid, hw in self.state.expert_to_hardware.items():
            by_hardware[hw.value].append(eid)

        # Load statistics
        ranking = self.load_history.get_load_ranking()
        load_stats = {
            "mean_ema_load": np.mean([x[1] for x in ranking]) if ranking else 0.0,
            "std_ema_load": np.std([x[1] for x in ranking]) if ranking else 0.0,
            "max_ema_load": ranking[0][1] if ranking else 0.0,
            "min_ema_load": ranking[-1][1] if ranking else 0.0,
        }

        return {
            "placement_version": self.state.placement_version,
            "by_hardware": dict(by_hardware),
            "hardware_counts": {k: len(v) for k, v in by_hardware.items()},
            "load_stats": load_stats,
            "last_rebalance_step": self.state.last_rebalance_step,
        }

    def save(self, path: str | Path) -> None:
        """Save placement state to file.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "num_experts": self.num_experts,
            "num_shared_experts": self.num_shared_experts,
            "config": {
                "hot_fraction": self.config.hot_fraction,
                "hot_expert_target": self.config.hot_expert_target.value,
                "cold_expert_target": self.config.cold_expert_target.value,
                "shared_expert_target": self.config.shared_expert_target.value,
                "ema_decay": self.config.ema_decay,
                "min_activations_for_placement": self.config.min_activations_for_placement,
                "rebalance_interval_steps": self.config.rebalance_interval_steps,
            },
            "state": self.state.to_dict(),
            "load_history": self.load_history.to_dict(),
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        LOGGER.info("Saved expert placement state to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> HeterogeneousExpertPlacement:
        """Load placement state from file.

        Args:
            path: Input file path

        Returns:
            Loaded HeterogeneousExpertPlacement instance
        """
        with open(path) as f:
            data = json.load(f)

        config = ExpertPlacementConfig(
            hot_fraction=data["config"]["hot_fraction"],
            hot_expert_target=HardwareTarget(data["config"]["hot_expert_target"]),
            cold_expert_target=HardwareTarget(data["config"]["cold_expert_target"]),
            shared_expert_target=HardwareTarget(data["config"]["shared_expert_target"]),
            ema_decay=data["config"]["ema_decay"],
            min_activations_for_placement=data["config"]["min_activations_for_placement"],
            rebalance_interval_steps=data["config"]["rebalance_interval_steps"],
        )

        placement = cls(
            num_experts=data["num_experts"],
            num_shared_experts=data["num_shared_experts"],
            config=config,
        )

        placement.state = ExpertPlacementState.from_dict(data["state"])
        placement.load_history = ExpertLoadHistory.from_dict(data["load_history"])

        LOGGER.info("Loaded expert placement state from %s", path)
        return placement


# =============================================================================
# Checkpoint Interoperability
# =============================================================================


class CheckpointFormat(Enum):
    """Checkpoint format types."""

    PYTORCH = "pytorch"  # Standard PyTorch state_dict
    CANDLE = "candle"  # Rust Candle format
    SAFETENSORS = "safetensors"  # SafeTensors format (shared)
    MLX = "mlx"  # Apple MLX format


@dataclass
class CheckpointInteropConfig:
    """Configuration for checkpoint interoperability."""

    # Name mapping between Candle and PyTorch
    enable_name_mapping: bool = True

    # Validation
    validate_checksums: bool = True
    validate_shapes: bool = True

    # Precision handling
    target_dtype: str = "float32"  # float16, bfloat16, float32

    # SafeTensors options
    use_safetensors: bool = True

    # Metadata preservation
    preserve_metadata: bool = True


# Name mapping from Candle (Rust) to PyTorch
CANDLE_TO_PYTORCH_NAME_MAP: dict[str, str] = {
    # Attention layers
    "attention.w_q": "self_attn.q_proj",
    "attention.w_k": "self_attn.k_proj",
    "attention.w_v": "self_attn.v_proj",
    "attention.w_o": "self_attn.o_proj",
    # MLA (Multi-head Latent Attention)
    "attention.kv_down": "self_attn.kv_down_proj",
    "attention.k_up": "self_attn.k_up_proj",
    "attention.v_up": "self_attn.v_up_proj",
    "attention.q_latent": "self_attn.q_latent_proj",
    # RoPE
    "attention.rope_k": "self_attn.rope_k",
    "attention.rope_q": "self_attn.rope_q",
    # MLP/FFN
    "mlp.gate": "mlp.gate_proj",
    "mlp.up": "mlp.up_proj",
    "mlp.down": "mlp.down_proj",
    # MoE
    "moe.gate": "moe.router",
    "moe.experts": "moe.experts",
    "moe.shared_experts": "moe.shared_experts",
    "moe.router_bias": "moe.router.bias",
    # Normalization
    "ln1": "input_layernorm",
    "ln2": "post_attention_layernorm",
    "final_ln": "model.norm",
    # Embeddings
    "embed_tokens": "model.embed_tokens",
    "lm_head": "lm_head",
}

# Inverse mapping
PYTORCH_TO_CANDLE_NAME_MAP: dict[str, str] = {v: k for k, v in CANDLE_TO_PYTORCH_NAME_MAP.items()}


@dataclass
class TensorMetadata:
    """Metadata for a checkpoint tensor."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    checksum: str = ""
    source_format: CheckpointFormat = CheckpointFormat.PYTORCH


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""

    version: int
    format: CheckpointFormat
    source_framework: str  # "pytorch", "candle", "mlx"
    timestamp: float
    tensors: dict[str, TensorMetadata] = field(default_factory=dict)
    training_step: int = 0
    model_config: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "version": self.version,
            "format": self.format.value,
            "source_framework": self.source_framework,
            "timestamp": self.timestamp,
            "tensors": {
                name: {
                    "name": meta.name,
                    "shape": list(meta.shape),
                    "dtype": meta.dtype,
                    "checksum": meta.checksum,
                    "source_format": meta.source_format.value,
                }
                for name, meta in self.tensors.items()
            },
            "training_step": self.training_step,
            "model_config": self.model_config,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointMetadata:
        """Deserialize from dictionary."""
        tensors = {}
        for name, meta_dict in data.get("tensors", {}).items():
            tensors[name] = TensorMetadata(
                name=meta_dict["name"],
                shape=tuple(meta_dict["shape"]),
                dtype=meta_dict["dtype"],
                checksum=meta_dict.get("checksum", ""),
                source_format=CheckpointFormat(meta_dict.get("source_format", "pytorch")),
            )

        return cls(
            version=data["version"],
            format=CheckpointFormat(data["format"]),
            source_framework=data["source_framework"],
            timestamp=data["timestamp"],
            tensors=tensors,
            training_step=data.get("training_step", 0),
            model_config=data.get("model_config", {}),
            extra=data.get("extra", {}),
        )


def _compute_tensor_checksum(data: bytes | np.ndarray) -> str:
    """Compute MD5 checksum of tensor data."""
    if isinstance(data, np.ndarray):
        data = data.tobytes()
    return hashlib.md5(data).hexdigest()


def _map_name_candle_to_pytorch(name: str) -> str:
    """Map Candle tensor name to PyTorch format.

    Args:
        name: Candle-style tensor name

    Returns:
        PyTorch-style tensor name
    """
    # Direct mapping
    if name in CANDLE_TO_PYTORCH_NAME_MAP:
        return CANDLE_TO_PYTORCH_NAME_MAP[name]

    # Pattern-based mapping for layers
    # e.g., "layers.0.attention.w_q" -> "model.layers.0.self_attn.q_proj"
    parts = name.split(".")

    mapped_parts = []
    i = 0
    while i < len(parts):
        part = parts[i]

        # Check if this is a layer index pattern
        if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            mapped_parts.append("model.layers")
            mapped_parts.append(parts[i + 1])
            i += 2
            continue

        # Map component names
        remaining = ".".join(parts[i:])
        for candle_pattern, pytorch_pattern in CANDLE_TO_PYTORCH_NAME_MAP.items():
            if remaining.startswith(candle_pattern):
                mapped_parts.append(pytorch_pattern)
                # Skip mapped parts
                skip_count = candle_pattern.count(".") + 1
                i += skip_count
                break
        else:
            # No mapping found, keep original
            mapped_parts.append(part)
            i += 1

    return ".".join(mapped_parts)


def _map_name_pytorch_to_candle(name: str) -> str:
    """Map PyTorch tensor name to Candle format.

    Args:
        name: PyTorch-style tensor name

    Returns:
        Candle-style tensor name
    """
    # Direct mapping
    if name in PYTORCH_TO_CANDLE_NAME_MAP:
        return PYTORCH_TO_CANDLE_NAME_MAP[name]

    # Pattern-based mapping
    parts = name.split(".")

    mapped_parts = []
    i = 0
    while i < len(parts):
        part = parts[i]

        # Handle "model.layers.N" -> "layers.N"
        if part == "model" and i + 1 < len(parts) and parts[i + 1] == "layers":
            # Skip "model"
            i += 1
            continue

        # Map component names
        remaining = ".".join(parts[i:])
        for pytorch_pattern, candle_pattern in PYTORCH_TO_CANDLE_NAME_MAP.items():
            if remaining.startswith(pytorch_pattern):
                mapped_parts.append(candle_pattern)
                skip_count = pytorch_pattern.count(".") + 1
                i += skip_count
                break
        else:
            mapped_parts.append(part)
            i += 1

    return ".".join(mapped_parts)


class CheckpointInterop:
    """Handles checkpoint interoperability between Rust (Candle) and PyTorch.

    This class provides:
    - Conversion between Candle and PyTorch checkpoint formats
    - Unified SafeTensors format for cross-framework compatibility
    - Name mapping between different naming conventions
    - Validation of checkpoint integrity

    Example:
        ```python
        interop = CheckpointInterop()

        # Convert Candle checkpoint to PyTorch
        pytorch_state = interop.convert_candle_to_pytorch(
            candle_path="checkpoints/rust/model.safetensors"
        )

        # Convert PyTorch checkpoint to Candle
        interop.convert_pytorch_to_candle(
            pytorch_path="checkpoints/pytorch/model.pt",
            output_path="checkpoints/rust/model.safetensors"
        )
        ```
    """

    def __init__(self, config: CheckpointInteropConfig | None = None):
        """Initialize checkpoint interop handler.

        Args:
            config: Interop configuration
        """
        self.config = config or CheckpointInteropConfig()

    def load_candle_checkpoint(
        self,
        path: str | Path,
    ) -> tuple[dict[str, np.ndarray], CheckpointMetadata]:
        """Load a Candle (Rust) checkpoint.

        Args:
            path: Path to checkpoint file (SafeTensors or numpy)

        Returns:
            Tuple of (tensor_dict, metadata)
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        tensors: dict[str, np.ndarray] = {}
        tensor_metadata: dict[str, TensorMetadata] = {}

        if path.suffix == ".safetensors" and SAFETENSORS_AVAILABLE:
            # Load SafeTensors
            with safe_open(str(path), framework="numpy") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    tensors[key] = tensor
                    tensor_metadata[key] = TensorMetadata(
                        name=key,
                        shape=tuple(tensor.shape),
                        dtype=str(tensor.dtype),
                        checksum=_compute_tensor_checksum(tensor),
                        source_format=CheckpointFormat.CANDLE,
                    )
        else:
            # Assume numpy format
            data = np.load(str(path), allow_pickle=True)
            for key in data.files:
                tensor = data[key]
                tensors[key] = tensor
                tensor_metadata[key] = TensorMetadata(
                    name=key,
                    shape=tuple(tensor.shape),
                    dtype=str(tensor.dtype),
                    checksum=_compute_tensor_checksum(tensor),
                    source_format=CheckpointFormat.CANDLE,
                )

        # Load or create metadata
        metadata_path = path.parent / f"{path.stem}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = CheckpointMetadata.from_dict(json.load(f))
                metadata.tensors = tensor_metadata
        else:
            metadata = CheckpointMetadata(
                version=1,
                format=CheckpointFormat.CANDLE,
                source_framework="candle",
                timestamp=time.time(),
                tensors=tensor_metadata,
            )

        LOGGER.info("Loaded Candle checkpoint from %s (%d tensors)", path, len(tensors))
        return tensors, metadata

    def convert_candle_to_pytorch(
        self,
        candle_path: str | Path,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Convert Candle checkpoint to PyTorch state_dict.

        Args:
            candle_path: Path to Candle checkpoint
            output_path: Optional path to save PyTorch checkpoint

        Returns:
            PyTorch state_dict
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for conversion")

        # Load Candle checkpoint
        tensors, metadata = self.load_candle_checkpoint(candle_path)

        # Convert to PyTorch tensors with name mapping
        pytorch_state: dict[str, Any] = {}

        for candle_name, np_tensor in tensors.items():
            # Map name
            if self.config.enable_name_mapping:
                pytorch_name = _map_name_candle_to_pytorch(candle_name)
            else:
                pytorch_name = candle_name

            # Convert to PyTorch tensor
            tensor = torch.from_numpy(np_tensor)

            # Convert dtype if needed
            if self.config.target_dtype == "float16":
                tensor = tensor.to(torch.float16)
            elif self.config.target_dtype == "bfloat16":
                tensor = tensor.to(torch.bfloat16)
            elif self.config.target_dtype == "float32":
                tensor = tensor.to(torch.float32)

            pytorch_state[pytorch_name] = tensor

        LOGGER.info(
            "Converted %d tensors from Candle to PyTorch format",
            len(pytorch_state),
        )

        # Save if output path provided
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if self.config.use_safetensors and SAFETENSORS_AVAILABLE:
                safetensors_save(pytorch_state, str(output_path))
            else:
                torch.save(pytorch_state, output_path)

            # Save metadata
            converted_metadata = CheckpointMetadata(
                version=metadata.version + 1,
                format=CheckpointFormat.PYTORCH,
                source_framework="pytorch",
                timestamp=time.time(),
                training_step=metadata.training_step,
                model_config=metadata.model_config,
                extra={
                    "converted_from": str(candle_path),
                    "original_format": "candle",
                },
            )

            metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(converted_metadata.to_dict(), f, indent=2)

            LOGGER.info("Saved PyTorch checkpoint to %s", output_path)

        return pytorch_state

    def convert_pytorch_to_candle(
        self,
        pytorch_path: str | Path,
        output_path: str | Path,
    ) -> dict[str, np.ndarray]:
        """Convert PyTorch checkpoint to Candle (SafeTensors) format.

        Args:
            pytorch_path: Path to PyTorch checkpoint
            output_path: Path to save Candle checkpoint

        Returns:
            Dictionary of numpy arrays
        """
        pytorch_path = Path(pytorch_path)
        output_path = Path(output_path)

        # Load PyTorch checkpoint
        if TORCH_AVAILABLE:
            state_dict = torch.load(pytorch_path, map_location="cpu")

            # Handle nested state_dict
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
        else:
            raise ImportError("PyTorch required to load PyTorch checkpoints")

        # Convert to numpy with name mapping
        candle_tensors: dict[str, np.ndarray] = {}

        for pytorch_name, tensor in state_dict.items():
            # Map name
            if self.config.enable_name_mapping:
                candle_name = _map_name_pytorch_to_candle(pytorch_name)
            else:
                candle_name = pytorch_name

            # Convert to numpy
            np_tensor = tensor.detach().cpu().numpy()

            # Convert dtype
            if self.config.target_dtype == "float32":
                np_tensor = np_tensor.astype(np.float32)
            elif self.config.target_dtype == "float16":
                np_tensor = np_tensor.astype(np.float16)

            candle_tensors[candle_name] = np_tensor

        LOGGER.info(
            "Converted %d tensors from PyTorch to Candle format",
            len(candle_tensors),
        )

        # Save as SafeTensors
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.config.use_safetensors and SAFETENSORS_AVAILABLE and TORCH_AVAILABLE:
            # Convert back to torch for safetensors
            torch_tensors = {k: torch.from_numpy(v) for k, v in candle_tensors.items()}
            safetensors_save(torch_tensors, str(output_path))
        else:
            # Save as numpy
            np.savez(str(output_path), **candle_tensors)

        # Save metadata
        metadata = CheckpointMetadata(
            version=1,
            format=CheckpointFormat.CANDLE,
            source_framework="candle",
            timestamp=time.time(),
            extra={
                "converted_from": str(pytorch_path),
                "original_format": "pytorch",
            },
        )

        metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        LOGGER.info("Saved Candle checkpoint to %s", output_path)

        return candle_tensors

    def validate_checkpoint(
        self,
        path: str | Path,
        expected_tensors: list[str] | None = None,
    ) -> dict[str, Any]:
        """Validate checkpoint integrity.

        Args:
            path: Path to checkpoint
            expected_tensors: Optional list of expected tensor names

        Returns:
            Validation results
        """
        path = Path(path)

        results: dict[str, Any] = {
            "valid": True,
            "path": str(path),
            "errors": [],
            "warnings": [],
            "tensor_count": 0,
        }

        if not path.exists():
            results["valid"] = False
            results["errors"].append(f"File not found: {path}")
            return results

        try:
            # Try to load
            if path.suffix == ".safetensors" and SAFETENSORS_AVAILABLE:
                with safe_open(str(path), framework="numpy") as f:
                    tensor_names = list(f.keys())
                    results["tensor_count"] = len(tensor_names)

                    # Check expected tensors
                    if expected_tensors:
                        missing = set(expected_tensors) - set(tensor_names)
                        if missing:
                            results["warnings"].append(f"Missing expected tensors: {missing}")

                        extra = set(tensor_names) - set(expected_tensors)
                        if extra:
                            results["warnings"].append(f"Extra tensors: {extra}")
            elif path.suffix == ".pt" and TORCH_AVAILABLE:
                state_dict = torch.load(path, map_location="cpu")
                if isinstance(state_dict, dict):
                    if "model_state_dict" in state_dict:
                        state_dict = state_dict["model_state_dict"]
                    results["tensor_count"] = len(state_dict)
                else:
                    results["warnings"].append("Unexpected checkpoint format")
            else:
                results["warnings"].append(f"Unknown file format: {path.suffix}")

        except Exception as e:
            results["valid"] = False
            results["errors"].append(f"Failed to load: {e}")

        return results

    def get_name_mapping_preview(
        self,
        path: str | Path,
        source_format: CheckpointFormat,
    ) -> list[tuple[str, str]]:
        """Preview name mapping that would be applied.

        Args:
            path: Path to checkpoint
            source_format: Source format (CANDLE or PYTORCH)

        Returns:
            List of (source_name, target_name) tuples
        """
        path = Path(path)

        # Get tensor names
        names: list[str] = []

        if path.suffix == ".safetensors" and SAFETENSORS_AVAILABLE:
            with safe_open(str(path), framework="numpy") as f:
                names = list(f.keys())
        elif path.suffix == ".pt" and TORCH_AVAILABLE:
            state_dict = torch.load(path, map_location="cpu")
            if isinstance(state_dict, dict):
                if "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
                names = list(state_dict.keys())

        # Apply mapping
        mapping: list[tuple[str, str]] = []

        if source_format == CheckpointFormat.CANDLE:
            for name in names:
                mapped = _map_name_candle_to_pytorch(name)
                mapping.append((name, mapped))
        else:
            for name in names:
                mapped = _map_name_pytorch_to_candle(name)
                mapping.append((name, mapped))

        return mapping


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Expert Placement
    "HardwareTarget",
    "ExpertLoadStats",
    "ExpertLoadHistory",
    "ExpertPlacementConfig",
    "ExpertPlacementState",
    "HeterogeneousExpertPlacement",
    # Checkpoint Interop
    "CheckpointFormat",
    "CheckpointInteropConfig",
    "CheckpointMetadata",
    "TensorMetadata",
    "CheckpointInterop",
    "CANDLE_TO_PYTORCH_NAME_MAP",
    "PYTORCH_TO_CANDLE_NAME_MAP",
]
