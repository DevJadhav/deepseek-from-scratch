"""GRPO alignment stage with framework selection and fallback support.

This stage implements Group Relative Policy Optimization (GRPO) for alignment
with automatic framework selection based on hardware capabilities and preferences.

Framework Selection Strategy (default):
- Generation/Rollout: MLX → Rust+Metal → PyTorch+MPS → CPU
- Policy Update/KL: PyTorch+CUDA → Rust+CUDA → MLX → CPU

Can be configured via:
1. Environment variable: DEEPSEEK_FRAMEWORK_PRESET=rust_primary
2. PipelineConfig.framework_preset
3. Direct FrameworkSelector injection
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import TYPE_CHECKING

from deepseek.pipeline.config import Backend, Stage
from deepseek.pipeline.framework_selector import (
    Framework,
    FrameworkSelector,
    PipelineFrameworkConfig,
    TaskType,
)
from deepseek.pipeline.runners import MLXRunner, PyTorchRunner, RustRunner
from deepseek.pipeline.stages.base import BaseStage, StageContext

if TYPE_CHECKING:
    from deepseek.pipeline.runners.base import BaseRunner

LOGGER = logging.getLogger(__name__)


class GRPOStage(BaseStage):
    """GRPO alignment stage with framework selection.

    Supports heterogeneous execution where generation happens on one
    framework (e.g., MLX on Apple Silicon) and training on another
    (e.g., PyTorch+CUDA).

    Configuration Options:
    - framework_selector: Injected FrameworkSelector instance
    - framework_preset: String preset ("default", "rust_primary", etc.)
    - fallback_enabled: Whether to enable automatic fallback (default: True)
    """

    stage_name = Stage.GRPO.value

    def __init__(
        self,
        config,
        framework_selector: FrameworkSelector | None = None,
        framework_preset: str | None = None,
    ):
        """Initialize GRPO stage.

        Args:
            config: Pipeline configuration
            framework_selector: Optional pre-configured selector
            framework_preset: Optional preset name for framework config
        """
        super().__init__(config)

        # Initialize framework selector
        if framework_selector is not None:
            self._framework_selector = framework_selector
        elif framework_preset:
            fw_config = PipelineFrameworkConfig.from_preset(framework_preset)
            self._framework_selector = FrameworkSelector(fw_config)
        else:
            # Use default heterogeneous configuration
            self._framework_selector = FrameworkSelector()

        self._log_framework_selection()

    def _log_framework_selection(self) -> None:
        """Log selected frameworks for each GRPO task."""
        gen_fw = self._framework_selector.select(TaskType.GENERATION)
        policy_fw = self._framework_selector.select(TaskType.POLICY_UPDATE)
        kl_fw = self._framework_selector.select(TaskType.KL_COMPUTATION)

        LOGGER.info(
            "GRPO framework selection: generation=%s, policy_update=%s, kl=%s",
            gen_fw.value,
            policy_fw.value,
            kl_fw.value,
        )

    def run(self, context: StageContext) -> StageContext:
        """Execute GRPO stage with framework selection.

        Uses heterogeneous execution:
        1. Generation/rollout uses selected generation framework
        2. Policy updates use selected training framework
        """
        dataset_uri = context.metadata.get(
            "grpo_dataset_path",
            context.metadata.get("sft_dataset_path", context.metadata.get("dataset_path")),
        )
        if not dataset_uri:
            raise RuntimeError("GRPOStage requires a dataset path in metadata")

        # Select frameworks for different GRPO tasks
        gen_framework = self._framework_selector.select(TaskType.GENERATION)
        policy_framework = self._framework_selector.select(TaskType.POLICY_UPDATE)

        # Get runner for policy updates (generation framework info passed via config)
        policy_runner = self._get_runner_for_framework(policy_framework)

        # Build training config
        training_overrides = asdict(self.config.training)
        training_overrides["learning_rate"] = self.config.grpo.learning_rate
        training_overrides["max_steps"] = self.config.grpo.num_iterations
        training_overrides["use_amp"] = False

        # Build extra config with framework info
        extra_config = {
            "stage": self.stage_name,
            "grpo_config": asdict(self.config.grpo),
            "reward_model_path": self.config.grpo.reward_model_path,
            "generation_framework": gen_framework.value,
            "policy_framework": policy_framework.value,
            "heterogeneous": gen_framework != policy_framework,
        }

        # For heterogeneous execution, use the policy runner as primary
        # but pass generation framework info for rollout phase
        result = policy_runner.run(
            dataset_uri=dataset_uri,
            pad_token_id=context.metadata.get("pad_token_id", 0),
            training_config=training_overrides,
            extra_config=extra_config,
        )

        context.previous_output = result.checkpoint_path
        context.metadata["grpo_checkpoint"] = result.checkpoint_path
        context.metadata["grpo_metrics"] = result.metrics
        context.metadata["grpo_generation_framework"] = gen_framework.value
        context.metadata["grpo_policy_framework"] = policy_framework.value
        return context

    def _get_runner_for_framework(self, framework: Framework) -> BaseRunner:
        """Get appropriate runner for the selected framework.

        Args:
            framework: Selected framework

        Returns:
            Runner instance for the framework
        """
        if framework.is_pytorch():
            return PyTorchRunner(self.config, stage=self.stage_name)
        if framework == Framework.MLX:
            return MLXRunner(self.config, stage=self.stage_name)
        if framework.is_rust():
            return RustRunner(self.config, stage=self.stage_name)

        # Fallback to PyTorch CPU
        LOGGER.warning("No specific runner for %s, falling back to PyTorch", framework.value)
        return PyTorchRunner(self.config, stage=self.stage_name)

    def _select_runner(self, backend: Backend) -> BaseRunner:
        """Legacy method for backward compatibility.

        Args:
            backend: Backend enum (deprecated)

        Returns:
            Runner instance
        """
        if backend in {Backend.PYTORCH_CUDA, Backend.PYTORCH_MPS, Backend.PYTORCH_CPU}:
            return PyTorchRunner(self.config, stage=self.stage_name)
        if backend == Backend.MLX:
            return MLXRunner(self.config, stage=self.stage_name)
        if backend == Backend.RUST:
            return RustRunner(self.config, stage=self.stage_name)
        raise NotImplementedError(f"Unsupported backend: {backend.value}")

    def get_framework_selector(self) -> FrameworkSelector:
        """Get the framework selector for inspection or modification.

        Returns:
            Current FrameworkSelector instance
        """
        return self._framework_selector

    def set_framework_preset(self, preset: str) -> None:
        """Change framework preset at runtime.

        Args:
            preset: Preset name ("default", "rust_primary", etc.)
        """
        fw_config = PipelineFrameworkConfig.from_preset(preset)
        self._framework_selector.reconfigure(fw_config)
        self._log_framework_selection()
