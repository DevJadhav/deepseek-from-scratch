"""Pre-training stage orchestrated via Ray runners."""

from __future__ import annotations

from typing import Optional

from ray_pipeline.config import Backend, Stage
from ray_pipeline.runners import MLXRunner, ModalRunner, PyTorchRunner, RustRunner
from ray_pipeline.stages.base import BaseStage, StageContext


class PretrainStage(BaseStage):
    stage_name = Stage.PRETRAIN.value

    def run(self, context: StageContext) -> StageContext:
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

        backend = self.config.detect_backend()
        runner = self._select_runner(backend)

        result = runner.run(
            dataset_uri=dataset_uri,
            pad_token_id=pad_token_id,
            extra_config={"stage": self.stage_name},
        )

        context.previous_output = result.checkpoint_path
        context.metadata["pretrain_checkpoint"] = result.checkpoint_path
        context.metadata["pretrain_metrics"] = result.metrics
        return context

    def _select_runner(self, backend: Backend):
        if backend in {Backend.PYTORCH_CUDA, Backend.PYTORCH_MPS, Backend.PYTORCH_CPU}:
            return PyTorchRunner(self.config, stage=self.stage_name)
        if backend == Backend.MLX:
            return MLXRunner(self.config, stage=self.stage_name)
        if backend == Backend.RUST:
            return RustRunner(self.config, stage=self.stage_name)
        if backend == Backend.MODAL_GPU:
            return ModalRunner(self.config, stage=self.stage_name)
        raise NotImplementedError(f"Unsupported backend: {backend.value}")
