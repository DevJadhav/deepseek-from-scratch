"""GRPO alignment stage."""

from __future__ import annotations

from dataclasses import asdict

from ray_pipeline.config import Backend, Stage
from ray_pipeline.runners import MLXRunner, PyTorchRunner, RustRunner
from ray_pipeline.stages.base import BaseStage, StageContext


class GRPOStage(BaseStage):
    stage_name = Stage.GRPO.value

    def run(self, context: StageContext) -> StageContext:
        dataset_uri = context.metadata.get(
            "grpo_dataset_path",
            context.metadata.get("sft_dataset_path", context.metadata.get("dataset_path")),
        )
        if not dataset_uri:
            raise RuntimeError("GRPOStage requires a dataset path in metadata")

        backend = self.config.detect_backend()
        runner = self._select_runner(backend)

        training_overrides = asdict(self.config.training)
        training_overrides["learning_rate"] = self.config.grpo.learning_rate
        training_overrides["max_steps"] = self.config.grpo.num_iterations
        training_overrides["use_amp"] = False

        result = runner.run(
            dataset_uri=dataset_uri,
            pad_token_id=context.metadata.get("pad_token_id", 0),
            training_config=training_overrides,
            extra_config={
                "stage": self.stage_name,
                "grpo_config": asdict(self.config.grpo),
                "reward_model_path": self.config.grpo.reward_model_path,
            },
        )

        context.previous_output = result.checkpoint_path
        context.metadata["grpo_checkpoint"] = result.checkpoint_path
        context.metadata["grpo_metrics"] = result.metrics
        return context

    def _select_runner(self, backend: Backend):
        if backend in {Backend.PYTORCH_CUDA, Backend.PYTORCH_MPS, Backend.PYTORCH_CPU}:
            return PyTorchRunner(self.config, stage=self.stage_name)
        if backend == Backend.MLX:
            return MLXRunner(self.config, stage=self.stage_name)
        if backend == Backend.RUST:
            return RustRunner(self.config, stage=self.stage_name)
        raise NotImplementedError(f"Unsupported backend: {backend.value}")
