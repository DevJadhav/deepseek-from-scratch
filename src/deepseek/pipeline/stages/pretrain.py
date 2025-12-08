"""Pre-training stage orchestrated via Ray runners with framework selection.

This stage implements pre-training with automatic framework selection based on
hardware capabilities and preferences.

Framework Selection Strategy:
- When backend=pytorch_*: PyTorch CUDA → MPS → CPU (NO Rust fallback)
- When backend=rust: Rust CUDA → Metal → CPU (NO PyTorch fallback)
- When backend=mlx: MLX → PyTorch MPS → CPU

Can be configured via:
1. PipelineConfig.backend (controls fallback chain)
2. Environment variable: DEEPSEEK_FRAMEWORK_PRESET=rust_primary
3. Direct FrameworkSelector injection
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from deepseek.pipeline.config import Backend, Stage
from deepseek.pipeline.framework_selector import (
    Framework,
    FrameworkSelector,
    PipelineFrameworkConfig,
    TaskType,
)
from deepseek.pipeline.runners import MLXRunner, ModalRunner, PyTorchRunner, RustRunner
from deepseek.pipeline.stages.base import BaseStage, StageContext

if TYPE_CHECKING:
    from deepseek.pipeline.runners.base import BaseRunner

LOGGER = logging.getLogger(__name__)


class PretrainStage(BaseStage):
    """Pre-training stage with framework selection.

    Supports automatic framework selection with fallback chain that
    respects the configured backend (no cross-backend fallback).

    Configuration Options:
    - config.backend: Controls which backend family to use (pytorch_*, rust, mlx)
    - framework_selector: Injected FrameworkSelector instance
    - framework_preset: String preset ("default", "rust_only", "pytorch_only")
    """

    stage_name = Stage.PRETRAIN.value

    def __init__(
        self,
        config,
        framework_selector: FrameworkSelector | None = None,
        framework_preset: str | None = None,
    ):
        """Initialize Pretrain stage.

        Args:
            config: Pipeline configuration
            framework_selector: Optional pre-configured selector
            framework_preset: Optional preset name for framework config
        """
        super().__init__(config)

        # Initialize framework selector - respect backend from config
        if framework_selector is not None:
            self._framework_selector = framework_selector
        else:
            # Use preset from argument, or derive from backend config
            preset = framework_preset or self.get_framework_preset()
            fw_config = PipelineFrameworkConfig.from_preset(preset)
            self._framework_selector = FrameworkSelector(fw_config)

        self._log_framework_selection()

    def _log_framework_selection(self) -> None:
        """Log selected frameworks for training tasks."""
        forward_fw = self._framework_selector.select(TaskType.FORWARD_PASS)
        backward_fw = self._framework_selector.select(TaskType.BACKWARD_PASS)
        opt_fw = self._framework_selector.select(TaskType.OPTIMIZER_STEP)

        LOGGER.info(
            "Pretrain framework selection: forward=%s, backward=%s, optimizer=%s",
            forward_fw.value,
            backward_fw.value,
            opt_fw.value,
        )

    def run(self, context: StageContext) -> StageContext:
        """Execute pre-training with framework selection."""
        dataset_uri = context.metadata.get("dataset_path")

        # Fall back to data_dir from config if no dataset_path in metadata
        if not dataset_uri:
            data_dir = self.config.data.data_dir
            if data_dir:
                import os

                # Check for stories/train structure (TinyStories dataset)
                stories_train_dir = os.path.join(data_dir, "stories", "train")
                train_dir = os.path.join(data_dir, "train")

                if os.path.exists(stories_train_dir):
                    dataset_uri = stories_train_dir
                    self.logger.info("Using stories/train from data_dir: %s", dataset_uri)
                elif os.path.exists(train_dir):
                    dataset_uri = train_dir
                    self.logger.info("Using data_dir/train from config: %s", dataset_uri)
                elif os.path.exists(data_dir):
                    dataset_uri = data_dir
                    self.logger.info("Using data_dir from config: %s", dataset_uri)

        if not dataset_uri:
            raise RuntimeError(
                "PretrainStage requires dataset_path in context metadata or data_dir in config. "
                "Run DataPrepStage first or set data.data_dir in config."
            )

        pad_token_id = context.metadata.get("pad_token_id", 0)

        # Select framework for forward pass (all training ops use same framework)
        training_framework = self._framework_selector.select(TaskType.FORWARD_PASS)
        runner = self._get_runner_for_framework(training_framework)

        result = runner.run(
            dataset_uri=dataset_uri,
            pad_token_id=pad_token_id,
            extra_config={
                "stage": self.stage_name,
                "framework": training_framework.value,
            },
        )

        context.previous_output = result.checkpoint_path
        context.metadata["pretrain_checkpoint"] = result.checkpoint_path
        context.metadata["pretrain_metrics"] = result.metrics
        context.metadata["pretrain_framework"] = training_framework.value
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
        if backend == Backend.MODAL_GPU:
            return ModalRunner(self.config, stage=self.stage_name)
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
