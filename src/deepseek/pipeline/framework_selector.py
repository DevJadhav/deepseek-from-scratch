"""
Framework Selection and Fallback System

This module provides a unified framework selection system that allows:
1. Configurable primary framework preferences
2. Automatic fallback to alternative frameworks when primary is unavailable
3. Interchangeable framework selection for different pipeline phases
4. Hardware detection and optimal framework selection

Framework Hierarchy Examples:
- Apple Silicon tasks: MLX (default) → Rust+Metal → PyTorch+MPS → CPU
- GPU training tasks: PyTorch+CUDA (default) → Rust+CUDA → Rust+Metal → CPU
- Data ingestion: Rust (default) → Python → CPU

Reference: production_hardening.md Section 3 (Full-Lifecycle Pipeline Audit)
"""

from __future__ import annotations

import logging
import os
import platform
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

LOGGER = logging.getLogger(__name__)


# =============================================================================
# Framework Definitions
# =============================================================================


class Framework(Enum):
    """Available compute frameworks."""

    # Python frameworks
    PYTHON_CPU = "python_cpu"
    PYTORCH_CPU = "pytorch_cpu"
    PYTORCH_CUDA = "pytorch_cuda"
    PYTORCH_MPS = "pytorch_mps"
    MLX = "mlx"

    # Rust frameworks
    RUST_CPU = "rust_cpu"
    RUST_CUDA = "rust_cuda"
    RUST_METAL = "rust_metal"

    # Special
    AUTO = "auto"  # Auto-detect best available

    def is_gpu(self) -> bool:
        """Check if framework uses GPU acceleration."""
        return self in {
            Framework.PYTORCH_CUDA,
            Framework.PYTORCH_MPS,
            Framework.MLX,
            Framework.RUST_CUDA,
            Framework.RUST_METAL,
        }

    def is_rust(self) -> bool:
        """Check if framework is Rust-based."""
        return self in {
            Framework.RUST_CPU,
            Framework.RUST_CUDA,
            Framework.RUST_METAL,
        }

    def is_pytorch(self) -> bool:
        """Check if framework is PyTorch-based."""
        return self in {
            Framework.PYTORCH_CPU,
            Framework.PYTORCH_CUDA,
            Framework.PYTORCH_MPS,
        }

    def is_apple_silicon(self) -> bool:
        """Check if framework is optimized for Apple Silicon."""
        return self in {
            Framework.MLX,
            Framework.RUST_METAL,
            Framework.PYTORCH_MPS,
        }


class TaskType(Enum):
    """Types of tasks in the training pipeline."""

    # Phase 1: Data Ingestion
    DATA_LOADING = "data_loading"
    DATA_PREPROCESSING = "data_preprocessing"
    TOKENIZATION = "tokenization"
    SHUFFLING = "shuffling"

    # Phase 2: Training
    FORWARD_PASS = "forward_pass"
    BACKWARD_PASS = "backward_pass"
    OPTIMIZER_STEP = "optimizer_step"
    GRADIENT_SYNC = "gradient_sync"
    EXPERT_ROUTING = "expert_routing"
    ATTENTION_COMPUTE = "attention_compute"
    CHECKPOINTING = "checkpointing"

    # Phase 3: Post-Training (GRPO)
    GENERATION = "generation"
    ROLLOUT = "rollout"
    REWARD_COMPUTATION = "reward_computation"
    POLICY_UPDATE = "policy_update"
    KL_COMPUTATION = "kl_computation"
    REFERENCE_MODEL = "reference_model"

    # General
    INFERENCE = "inference"
    EXPORT = "export"


# =============================================================================
# Framework Availability Detection
# =============================================================================


@dataclass
class FrameworkAvailability:
    """Tracks which frameworks are available on the current system."""

    # Python/PyTorch
    pytorch_available: bool = False
    pytorch_cuda_available: bool = False
    pytorch_mps_available: bool = False
    cuda_device_count: int = 0
    cuda_compute_capability: float = 0.0

    # MLX
    mlx_available: bool = False

    # Rust
    rust_available: bool = False
    rust_cuda_available: bool = False
    rust_metal_available: bool = False

    # System info
    is_apple_silicon: bool = False
    cpu_count: int = 1
    memory_gb: float = 8.0

    @classmethod
    def detect(cls) -> FrameworkAvailability:
        """Auto-detect available frameworks."""
        availability = cls()

        # Detect system
        availability.is_apple_silicon = (
            platform.system() == "Darwin" and platform.machine().startswith("arm")
        )
        availability.cpu_count = os.cpu_count() or 1

        # Detect memory
        try:
            import psutil

            availability.memory_gb = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            availability.memory_gb = 8.0

        # Detect PyTorch
        try:
            import torch

            availability.pytorch_available = True
            availability.pytorch_cuda_available = torch.cuda.is_available()
            availability.pytorch_mps_available = (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            )

            if availability.pytorch_cuda_available:
                availability.cuda_device_count = torch.cuda.device_count()
                if availability.cuda_device_count > 0:
                    props = torch.cuda.get_device_properties(0)
                    availability.cuda_compute_capability = float(f"{props.major}.{props.minor}")
        except ImportError:
            pass

        # Detect MLX
        try:
            import mlx.core  # noqa: F401

            availability.mlx_available = True
        except ImportError:
            pass

        # Detect Rust (check if library is available)
        try:
            # Try to import the Rust Python bindings
            from deepseek_rust import is_available  # noqa: F401

            availability.rust_available = True
            # Check Rust CUDA/Metal support
            try:
                from deepseek_rust import has_cuda, has_metal

                availability.rust_cuda_available = has_cuda()
                availability.rust_metal_available = has_metal()
            except ImportError:
                # Infer from system
                availability.rust_cuda_available = availability.pytorch_cuda_available
                availability.rust_metal_available = availability.is_apple_silicon
        except ImportError:
            # Rust bindings not available, but we can still use subprocess
            availability.rust_available = _check_rust_binary_available()
            availability.rust_cuda_available = availability.pytorch_cuda_available
            availability.rust_metal_available = availability.is_apple_silicon

        return availability

    def get_available_frameworks(self) -> list[Framework]:
        """Get list of all available frameworks."""
        available = [Framework.PYTHON_CPU]

        if self.pytorch_available:
            available.append(Framework.PYTORCH_CPU)
            if self.pytorch_cuda_available:
                available.append(Framework.PYTORCH_CUDA)
            if self.pytorch_mps_available:
                available.append(Framework.PYTORCH_MPS)

        if self.mlx_available:
            available.append(Framework.MLX)

        if self.rust_available:
            available.append(Framework.RUST_CPU)
            if self.rust_cuda_available:
                available.append(Framework.RUST_CUDA)
            if self.rust_metal_available:
                available.append(Framework.RUST_METAL)

        return available

    def is_framework_available(self, framework: Framework) -> bool:
        """Check if a specific framework is available."""
        if framework == Framework.AUTO:
            return True

        mapping = {
            Framework.PYTHON_CPU: True,
            Framework.PYTORCH_CPU: self.pytorch_available,
            Framework.PYTORCH_CUDA: self.pytorch_cuda_available,
            Framework.PYTORCH_MPS: self.pytorch_mps_available,
            Framework.MLX: self.mlx_available,
            Framework.RUST_CPU: self.rust_available,
            Framework.RUST_CUDA: self.rust_cuda_available,
            Framework.RUST_METAL: self.rust_metal_available,
        }
        return mapping.get(framework, False)


def _check_rust_binary_available() -> bool:
    """Check if Rust binary is available."""
    import shutil

    return shutil.which("deepseek-rust") is not None


# =============================================================================
# Framework Selection Configuration
# =============================================================================


@dataclass
class FrameworkPreference:
    """Defines framework preference with fallback chain."""

    primary: Framework
    fallbacks: list[Framework] = field(default_factory=list)
    name: str = ""

    def get_chain(self) -> list[Framework]:
        """Get full fallback chain starting with primary."""
        return [self.primary, *self.fallbacks]

    @classmethod
    def pytorch_only(cls, name: str = "pytorch_only") -> FrameworkPreference:
        """PyTorch-only preference chain (NO cross-backend fallback).
        
        Fallback: CUDA → MPS → CPU (all within PyTorch)
        """
        return cls(
            primary=Framework.PYTORCH_CUDA,
            fallbacks=[
                Framework.PYTORCH_MPS,
                Framework.PYTORCH_CPU,
            ],
            name=name,
        )

    @classmethod
    def pytorch_mps_primary(cls, name: str = "pytorch_mps") -> FrameworkPreference:
        """PyTorch MPS-primary preference chain (NO cross-backend fallback).
        
        Fallback: MPS → CPU (all within PyTorch)
        """
        return cls(
            primary=Framework.PYTORCH_MPS,
            fallbacks=[
                Framework.PYTORCH_CPU,
            ],
            name=name,
        )

    @classmethod
    def rust_only(cls, name: str = "rust_only") -> FrameworkPreference:
        """Rust-only preference chain (NO cross-backend fallback).
        
        Fallback: CUDA → Metal → CPU (all within Rust)
        """
        return cls(
            primary=Framework.RUST_CUDA,
            fallbacks=[
                Framework.RUST_METAL,
                Framework.RUST_CPU,
            ],
            name=name,
        )

    @classmethod
    def rust_metal_only(cls, name: str = "rust_metal_only") -> FrameworkPreference:
        """Rust Metal-primary preference chain (NO cross-backend fallback).
        
        Fallback: Metal → CPU (all within Rust)
        """
        return cls(
            primary=Framework.RUST_METAL,
            fallbacks=[
                Framework.RUST_CPU,
            ],
            name=name,
        )

    @classmethod
    def apple_silicon_optimized(cls, name: str = "apple_silicon") -> FrameworkPreference:
        """Preference chain for Apple Silicon tasks.

        Default: MLX → Rust+Metal → PyTorch+MPS → CPU
        """
        return cls(
            primary=Framework.MLX,
            fallbacks=[
                Framework.RUST_METAL,
                Framework.PYTORCH_MPS,
                Framework.RUST_CPU,
                Framework.PYTORCH_CPU,
            ],
            name=name,
        )

    @classmethod
    def rust_metal_primary(cls, name: str = "rust_metal") -> FrameworkPreference:
        """Preference chain with Rust+Metal as primary.

        Default: Rust+Metal → MLX → PyTorch+MPS → CPU
        """
        return cls(
            primary=Framework.RUST_METAL,
            fallbacks=[
                Framework.MLX,
                Framework.PYTORCH_MPS,
                Framework.RUST_CPU,
                Framework.PYTORCH_CPU,
            ],
            name=name,
        )

    @classmethod
    def gpu_training(cls, name: str = "gpu_training") -> FrameworkPreference:
        """Preference chain for GPU training tasks.

        Default: PyTorch+CUDA → Rust+CUDA → Rust+Metal → MLX → CPU
        """
        return cls(
            primary=Framework.PYTORCH_CUDA,
            fallbacks=[
                Framework.RUST_CUDA,
                Framework.RUST_METAL,
                Framework.MLX,
                Framework.PYTORCH_MPS,
                Framework.PYTORCH_CPU,
            ],
            name=name,
        )

    @classmethod
    def rust_gpu_primary(cls, name: str = "rust_gpu") -> FrameworkPreference:
        """Preference chain with Rust+GPU as primary.

        Default: Rust+CUDA → Rust+Metal → PyTorch+CUDA → MLX → CPU
        """
        return cls(
            primary=Framework.RUST_CUDA,
            fallbacks=[
                Framework.RUST_METAL,
                Framework.PYTORCH_CUDA,
                Framework.MLX,
                Framework.PYTORCH_MPS,
                Framework.RUST_CPU,
            ],
            name=name,
        )

    @classmethod
    def data_processing(cls, name: str = "data_processing") -> FrameworkPreference:
        """Preference chain for data processing tasks.

        Default: Rust+CPU → Python+CPU (I/O bound, no GPU needed)
        """
        return cls(
            primary=Framework.RUST_CPU,
            fallbacks=[Framework.PYTHON_CPU, Framework.PYTORCH_CPU],
            name=name,
        )


@dataclass
class PipelineFrameworkConfig:
    """Configuration for framework selection across pipeline phases."""

    # Phase 1: Data Ingestion
    data_loading: FrameworkPreference = field(
        default_factory=lambda: FrameworkPreference.data_processing("data_loading")
    )
    tokenization: FrameworkPreference = field(
        default_factory=lambda: FrameworkPreference.data_processing("tokenization")
    )
    shuffling: FrameworkPreference = field(
        default_factory=lambda: FrameworkPreference.data_processing("shuffling")
    )

    # Phase 2: Training (MoE Loop)
    forward_pass: FrameworkPreference = field(
        default_factory=lambda: FrameworkPreference.gpu_training("forward_pass")
    )
    backward_pass: FrameworkPreference = field(
        default_factory=lambda: FrameworkPreference.gpu_training("backward_pass")
    )
    optimizer_step: FrameworkPreference = field(
        default_factory=lambda: FrameworkPreference.gpu_training("optimizer_step")
    )
    expert_routing: FrameworkPreference = field(
        default_factory=lambda: FrameworkPreference.gpu_training("expert_routing")
    )
    attention_compute: FrameworkPreference = field(
        default_factory=lambda: FrameworkPreference.gpu_training("attention_compute")
    )
    checkpointing: FrameworkPreference = field(
        default_factory=lambda: FrameworkPreference.data_processing("checkpointing")
    )

    # Phase 3: Post-Training (GRPO)
    generation: FrameworkPreference = field(
        default_factory=lambda: FrameworkPreference.apple_silicon_optimized("generation")
    )
    rollout: FrameworkPreference = field(
        default_factory=lambda: FrameworkPreference.apple_silicon_optimized("rollout")
    )
    reward_computation: FrameworkPreference = field(
        default_factory=lambda: FrameworkPreference.gpu_training("reward_computation")
    )
    policy_update: FrameworkPreference = field(
        default_factory=lambda: FrameworkPreference.gpu_training("policy_update")
    )
    kl_computation: FrameworkPreference = field(
        default_factory=lambda: FrameworkPreference.gpu_training("kl_computation")
    )
    reference_model: FrameworkPreference = field(
        default_factory=lambda: FrameworkPreference.apple_silicon_optimized("reference_model")
    )

    # General
    inference: FrameworkPreference = field(
        default_factory=lambda: FrameworkPreference.apple_silicon_optimized("inference")
    )
    export: FrameworkPreference = field(
        default_factory=lambda: FrameworkPreference.data_processing("export")
    )

    def get_preference(self, task_type: TaskType) -> FrameworkPreference:
        """Get framework preference for a task type."""
        mapping = {
            TaskType.DATA_LOADING: self.data_loading,
            TaskType.DATA_PREPROCESSING: self.data_loading,
            TaskType.TOKENIZATION: self.tokenization,
            TaskType.SHUFFLING: self.shuffling,
            TaskType.FORWARD_PASS: self.forward_pass,
            TaskType.BACKWARD_PASS: self.backward_pass,
            TaskType.OPTIMIZER_STEP: self.optimizer_step,
            TaskType.GRADIENT_SYNC: self.optimizer_step,
            TaskType.EXPERT_ROUTING: self.expert_routing,
            TaskType.ATTENTION_COMPUTE: self.attention_compute,
            TaskType.CHECKPOINTING: self.checkpointing,
            TaskType.GENERATION: self.generation,
            TaskType.ROLLOUT: self.rollout,
            TaskType.REWARD_COMPUTATION: self.reward_computation,
            TaskType.POLICY_UPDATE: self.policy_update,
            TaskType.KL_COMPUTATION: self.kl_computation,
            TaskType.REFERENCE_MODEL: self.reference_model,
            TaskType.INFERENCE: self.inference,
            TaskType.EXPORT: self.export,
        }
        return mapping.get(task_type, FrameworkPreference.gpu_training(task_type.value))

    def set_all_to_rust_primary(self) -> None:
        """Set all GPU tasks to use Rust as primary framework (NO cross-backend fallback)."""
        # Training tasks use Rust-only fallback chain
        rust_only = FrameworkPreference.rust_only()
        rust_metal_only = FrameworkPreference.rust_metal_only()
        
        self.forward_pass = rust_only
        self.backward_pass = rust_only
        self.optimizer_step = rust_only
        self.expert_routing = rust_only
        self.attention_compute = rust_only

        # GRPO generation uses Rust+Metal only
        self.generation = rust_metal_only
        self.rollout = rust_metal_only
        self.reference_model = rust_metal_only

        # Policy update uses Rust only
        self.policy_update = rust_only
        self.kl_computation = rust_only
        self.reward_computation = rust_only
        
        # Data processing stays Rust
        self.data_loading = FrameworkPreference(
            primary=Framework.RUST_CPU, 
            fallbacks=[Framework.PYTHON_CPU],
            name="data_loading_rust"
        )
        self.tokenization = FrameworkPreference(
            primary=Framework.RUST_CPU,
            fallbacks=[Framework.PYTHON_CPU],
            name="tokenization_rust"
        )
        self.checkpointing = FrameworkPreference(
            primary=Framework.RUST_CPU,
            fallbacks=[Framework.PYTHON_CPU],
            name="checkpointing_rust"
        )
        self.export = FrameworkPreference(
            primary=Framework.RUST_CPU,
            fallbacks=[Framework.PYTHON_CPU],
            name="export_rust"
        )

    def set_all_to_pytorch_primary(self) -> None:
        """Set all tasks to use PyTorch as primary framework (NO cross-backend fallback)."""
        # Training tasks use PyTorch-only fallback chain
        pytorch_only = FrameworkPreference.pytorch_only()
        pytorch_mps = FrameworkPreference.pytorch_mps_primary()
        
        self.forward_pass = pytorch_only
        self.backward_pass = pytorch_only
        self.optimizer_step = pytorch_only
        self.expert_routing = pytorch_only
        self.attention_compute = pytorch_only

        # GRPO tasks use PyTorch MPS primary (for Apple Silicon)
        self.generation = pytorch_mps
        self.rollout = pytorch_mps
        self.reference_model = pytorch_mps
        self.policy_update = pytorch_only
        self.kl_computation = pytorch_only
        self.reward_computation = pytorch_only
        
        # Inference uses PyTorch MPS
        self.inference = pytorch_mps
        
        # Data processing uses Python/PyTorch CPU
        python_cpu = FrameworkPreference(
            primary=Framework.PYTORCH_CPU,
            fallbacks=[Framework.PYTHON_CPU],
            name="data_pytorch"
        )
        self.data_loading = python_cpu
        self.tokenization = python_cpu
        self.checkpointing = python_cpu
        self.export = python_cpu

    @classmethod
    def from_preset(cls, preset: str) -> PipelineFrameworkConfig:
        """Create configuration from preset name.

        Presets:
        - "default": Standard configuration (PyTorch+CUDA training, MLX generation)
        - "rust_primary": Rust as primary for all tasks (NO cross-backend fallback)
        - "rust_only": Rust-only, no PyTorch fallback 
        - "pytorch_only": PyTorch for everything (NO cross-backend fallback)
        - "pytorch_mps": PyTorch MPS primary (for Apple Silicon, NO cross-backend fallback)
        - "apple_silicon": Optimized for Apple Silicon machines
        - "heterogeneous": Mixed MLX generation + PyTorch training
        """
        config = cls()

        if preset in ("rust_primary", "rust_only"):
            config.set_all_to_rust_primary()
        elif preset in ("pytorch_only", "pytorch_mps"):
            config.set_all_to_pytorch_primary()
        elif preset == "apple_silicon":
            # Everything on Apple Silicon
            apple_pref = FrameworkPreference.apple_silicon_optimized()
            rust_metal = FrameworkPreference.rust_metal_primary()
            config.forward_pass = apple_pref
            config.backward_pass = apple_pref
            config.optimizer_step = apple_pref
            config.generation = rust_metal
            config.rollout = rust_metal
            config.policy_update = apple_pref
        elif preset == "heterogeneous":
            # Default heterogeneous: MLX generation, PyTorch training
            pass  # Default configuration is already heterogeneous

        return config


# =============================================================================
# Framework Selector
# =============================================================================


class FrameworkSelector:
    """Selects the best available framework based on preferences and availability."""

    def __init__(
        self,
        config: PipelineFrameworkConfig | None = None,
        availability: FrameworkAvailability | None = None,
    ):
        """Initialize framework selector.

        Args:
            config: Framework preference configuration
            availability: Pre-detected availability, or None to auto-detect
        """
        self.config = config or PipelineFrameworkConfig()
        self.availability = availability or FrameworkAvailability.detect()
        self._selection_cache: dict[TaskType, Framework] = {}

        LOGGER.info(
            "FrameworkSelector initialized. Available: %s",
            [f.value for f in self.availability.get_available_frameworks()],
        )

    def select(self, task_type: TaskType) -> Framework:
        """Select the best framework for a task type.

        Iterates through the preference chain and returns the first
        available framework.

        Args:
            task_type: Type of task to run

        Returns:
            Best available framework for the task
        """
        # Check cache
        if task_type in self._selection_cache:
            return self._selection_cache[task_type]

        preference = self.config.get_preference(task_type)
        selected = self._select_from_preference(preference)

        # Cache result
        self._selection_cache[task_type] = selected

        LOGGER.debug(
            "Selected framework for %s: %s (preference chain: %s)",
            task_type.value,
            selected.value,
            [f.value for f in preference.get_chain()],
        )

        return selected

    def _select_from_preference(self, preference: FrameworkPreference) -> Framework:
        """Select first available framework from preference chain."""
        for framework in preference.get_chain():
            if self.availability.is_framework_available(framework):
                return framework

        # Fallback to CPU
        if self.availability.pytorch_available:
            return Framework.PYTORCH_CPU
        return Framework.PYTHON_CPU

    def select_with_override(
        self, task_type: TaskType, override: Framework | None = None
    ) -> Framework:
        """Select framework with optional override.

        Args:
            task_type: Type of task
            override: Framework to use (overrides preference)

        Returns:
            Selected framework
        """
        if override is not None and override != Framework.AUTO:
            if self.availability.is_framework_available(override):
                return override
            LOGGER.warning(
                "Requested framework %s not available, falling back to preference",
                override.value,
            )

        return self.select(task_type)

    def get_executor(
        self, task_type: TaskType, override: Framework | None = None
    ) -> FrameworkExecutor:
        """Get executor for a task type.

        Args:
            task_type: Type of task
            override: Framework to use (overrides preference)

        Returns:
            FrameworkExecutor configured for the selected framework
        """
        framework = self.select_with_override(task_type, override)
        return FrameworkExecutor(framework, task_type)

    def clear_cache(self) -> None:
        """Clear selection cache (useful after reconfiguration)."""
        self._selection_cache.clear()

    def reconfigure(self, config: PipelineFrameworkConfig) -> None:
        """Update configuration and clear cache."""
        self.config = config
        self.clear_cache()


# =============================================================================
# Framework Executor
# =============================================================================

T = TypeVar("T")


class FrameworkExecutor:
    """Executes tasks using the selected framework."""

    def __init__(self, framework: Framework, task_type: TaskType):
        """Initialize executor.

        Args:
            framework: Framework to use
            task_type: Type of task (for logging)
        """
        self.framework = framework
        self.task_type = task_type

    def execute(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute a function using the selected framework.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result of function execution
        """
        LOGGER.debug("Executing %s using %s", self.task_type.value, self.framework.value)

        # Set up framework context
        self._setup_context()

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            self._cleanup_context()

    def _setup_context(self) -> None:
        """Set up framework-specific context."""
        if self.framework.is_pytorch():
            self._setup_pytorch_context()
        elif self.framework == Framework.MLX:
            self._setup_mlx_context()
        elif self.framework.is_rust():
            self._setup_rust_context()

    def _cleanup_context(self) -> None:
        """Clean up framework-specific context."""
        pass  # Cleanup if needed

    def _setup_pytorch_context(self) -> None:
        """Set up PyTorch context."""
        import torch

        if self.framework == Framework.PYTORCH_CUDA:
            if torch.cuda.is_available():
                torch.cuda.set_device(0)
        elif self.framework == Framework.PYTORCH_MPS:
            # MPS is auto-selected when available
            pass

    def _setup_mlx_context(self) -> None:
        """Set up MLX context."""
        try:
            import mlx.core as mx

            mx.set_default_device(mx.gpu)
        except ImportError:
            pass

    def _setup_rust_context(self) -> None:
        """Set up Rust context."""
        # Rust context is set up via environment variables
        if self.framework == Framework.RUST_CUDA:
            os.environ["DEEPSEEK_RUST_BACKEND"] = "cuda"
        elif self.framework == Framework.RUST_METAL:
            os.environ["DEEPSEEK_RUST_BACKEND"] = "metal"
        else:
            os.environ["DEEPSEEK_RUST_BACKEND"] = "cpu"

    def get_device_string(self) -> str:
        """Get device string for the framework."""
        device_map = {
            Framework.PYTHON_CPU: "cpu",
            Framework.PYTORCH_CPU: "cpu",
            Framework.PYTORCH_CUDA: "cuda",
            Framework.PYTORCH_MPS: "mps",
            Framework.MLX: "gpu",  # MLX uses "gpu" for Apple Silicon
            Framework.RUST_CPU: "cpu",
            Framework.RUST_CUDA: "cuda",
            Framework.RUST_METAL: "metal",
        }
        return device_map.get(self.framework, "cpu")


# =============================================================================
# Global Singleton
# =============================================================================

_global_selector: FrameworkSelector | None = None


def get_framework_selector() -> FrameworkSelector:
    """Get or create the global framework selector."""
    global _global_selector
    if _global_selector is None:
        _global_selector = FrameworkSelector()
    return _global_selector


def configure_framework_selector(
    config: PipelineFrameworkConfig | None = None,
    preset: str | None = None,
) -> FrameworkSelector:
    """Configure the global framework selector.

    Args:
        config: Framework preference configuration
        preset: Preset name ("default", "rust_primary", "pytorch_only", etc.)

    Returns:
        Configured FrameworkSelector
    """
    global _global_selector

    if preset:
        config = PipelineFrameworkConfig.from_preset(preset)

    _global_selector = FrameworkSelector(config=config)
    return _global_selector


def select_framework(task_type: TaskType, override: Framework | None = None) -> Framework:
    """Select framework for a task type using global selector.

    Args:
        task_type: Type of task
        override: Optional framework override

    Returns:
        Selected framework
    """
    return get_framework_selector().select_with_override(task_type, override)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    "Framework",
    "TaskType",
    # Dataclasses
    "FrameworkAvailability",
    "FrameworkPreference",
    "PipelineFrameworkConfig",
    # Classes
    "FrameworkSelector",
    "FrameworkExecutor",
    # Functions
    "get_framework_selector",
    "configure_framework_selector",
    "select_framework",
]
