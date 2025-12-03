"""Knowledge distillation stage."""

from __future__ import annotations

from dataclasses import asdict

from ray_pipeline.config import Backend, Stage
from ray_pipeline.runners import MLXRunner, PyTorchRunner, RustRunner
from ray_pipeline.stages.base import BaseStage, StageContext


class DistillationStage(BaseStage):
    stage_name = Stage.DISTILLATION.value

    def run(self, context: StageContext) -> StageContext:
        dataset_uri = context.metadata.get(
            "distillation_dataset_path",
            context.metadata.get("dataset_path"),
        )
        if not dataset_uri:
            raise RuntimeError("DistillationStage requires a dataset path")

        teacher_ckpt = self.config.distillation.teacher_model_path or context.metadata.get(
            "teacher_checkpoint"
        )
        if not teacher_ckpt:
            self.logger.warning(
                "Teacher checkpoint not provided. Distillation will fallback to standard training"
            )

        backend = self.config.detect_backend()
        runner = self._select_runner(backend)

        training_overrides = asdict(self.config.training)
        training_overrides["learning_rate"] = self.config.distillation.learning_rate
        training_overrides["max_steps"] = max(
            training_overrides["max_steps"], self.config.distillation.num_epochs * 1000
        )
        training_overrides["use_amp"] = True

        result = runner.run(
            dataset_uri=dataset_uri,
            pad_token_id=context.metadata.get("pad_token_id", 0),
            training_config=training_overrides,
            extra_config={
                "stage": self.stage_name,
                "distillation": asdict(self.config.distillation),
                "teacher_checkpoint": teacher_ckpt,
            },
        )

        context.previous_output = result.checkpoint_path
        context.metadata["distillation_checkpoint"] = result.checkpoint_path
        context.metadata["distillation_metrics"] = result.metrics
        return context

    def _select_runner(self, backend: Backend):
        if backend in {Backend.PYTORCH_CUDA, Backend.PYTORCH_MPS, Backend.PYTORCH_CPU}:
            return PyTorchRunner(self.config, stage=self.stage_name)
        if backend == Backend.MLX:
            return MLXRunner(self.config, stage=self.stage_name)
        if backend == Backend.RUST:
            return RustRunner(self.config, stage=self.stage_name)
        raise NotImplementedError(f"Unsupported backend: {backend.value}")
