"""Model export stage."""

from __future__ import annotations

import shutil
from pathlib import Path

from ray_pipeline.config import Stage
from ray_pipeline.stages.base import BaseStage, StageContext


class ExportStage(BaseStage):
    stage_name = Stage.EXPORT.value

    def run(self, context: StageContext) -> StageContext:
        checkpoint = (
            context.metadata.get("distillation_checkpoint")
            or context.metadata.get("grpo_checkpoint")
            or context.metadata.get("sft_checkpoint")
            or context.metadata.get("pretrain_checkpoint")
            or context.previous_output
        )
        if not checkpoint:
            raise RuntimeError("ExportStage requires a checkpoint to export")

        output_dir = Path(self.config.export.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        export_path = output_dir / f"{self.config.run_name}-final.ckpt"
        self.logger.info("Copying checkpoint %s -> %s", checkpoint, export_path)

        try:
            shutil.copy(checkpoint, export_path)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Could not copy checkpoint directly: %s", exc)

        context.metadata["export_path"] = str(export_path)
        context.previous_output = str(export_path)
        return context
