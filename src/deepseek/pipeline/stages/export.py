"""Model export stage."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from deepseek.pipeline.config import Stage
from deepseek.pipeline.stages.base import BaseStage, StageContext


class ExportStage(BaseStage):
    """
    Export stage for converting trained models to deployable formats.
    
    Supports:
    - Checkpoint copying to final location
    - GGUF export for llama.cpp/ollama
    - Safetensors format
    """
    
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

        # Copy checkpoint to final location
        export_path = output_dir / f"{self.config.run_name}-final.ckpt"
        self.logger.info("Copying checkpoint %s -> %s", checkpoint, export_path)

        try:
            if Path(checkpoint).is_dir():
                shutil.copytree(checkpoint, export_path, dirs_exist_ok=True)
            else:
                shutil.copy(checkpoint, export_path)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Could not copy checkpoint directly: %s", exc)

        context.metadata["export_path"] = str(export_path)

        # Export to GGUF if configured
        if getattr(self.config.export, "export_gguf", False):
            gguf_path = self._export_gguf(checkpoint, output_dir)
            if gguf_path:
                context.metadata["gguf_path"] = str(gguf_path)

        # Export to safetensors if configured
        if getattr(self.config.export, "export_safetensors", True):
            safetensors_path = self._export_safetensors(checkpoint, output_dir)
            if safetensors_path:
                context.metadata["safetensors_path"] = str(safetensors_path)

        # Save export metadata
        metadata_path = output_dir / "export_metadata.json"
        export_metadata = {
            "run_name": self.config.run_name,
            "source_checkpoint": str(checkpoint),
            "export_path": str(export_path),
            "export_gguf": getattr(self.config.export, "export_gguf", False),
            "export_safetensors": getattr(self.config.export, "export_safetensors", True),
            "model_size": getattr(self.config.model, "size", "unknown"),
        }
        
        if "gguf_path" in context.metadata:
            export_metadata["gguf_path"] = context.metadata["gguf_path"]
        if "safetensors_path" in context.metadata:
            export_metadata["safetensors_path"] = context.metadata["safetensors_path"]
        
        with open(metadata_path, "w") as f:
            json.dump(export_metadata, f, indent=2)
        
        context.previous_output = str(export_path)
        self.logger.info("Export complete. Metadata saved to %s", metadata_path)
        
        return context
    
    def _export_gguf(self, checkpoint: str, output_dir: Path) -> Path | None:
        """Export model to GGUF format."""
        gguf_path = output_dir / f"{self.config.run_name}.gguf"
        quantization = getattr(self.config.export, "quantization", "q8_0")
        
        self.logger.info("Exporting to GGUF format: %s (quantization: %s)", 
                         gguf_path, quantization)
        
        try:
            # Find the export script
            project_root = Path(__file__).resolve().parents[2]
            export_script = project_root / "scripts" / "export_gguf.py"
            
            if not export_script.exists():
                self.logger.warning("GGUF export script not found: %s", export_script)
                return None
            
            # Run export
            result = subprocess.run(
                [
                    sys.executable, str(export_script),
                    "--checkpoint", str(checkpoint),
                    "--output", str(gguf_path),
                    "--quantization", quantization,
                    "--model-name", self.config.run_name,
                ],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )
            
            if result.returncode == 0:
                self.logger.info("GGUF export successful: %s", gguf_path)
                return gguf_path
            else:
                self.logger.warning("GGUF export failed: %s", result.stderr)
                return None
                
        except subprocess.TimeoutExpired:
            self.logger.warning("GGUF export timed out")
            return None
        except Exception as exc:
            self.logger.warning("GGUF export error: %s", exc)
            return None
    
    def _export_safetensors(self, checkpoint: str, output_dir: Path) -> Path | None:
        """Export model to safetensors format if not already."""
        safetensors_path = output_dir / f"{self.config.run_name}.safetensors"
        checkpoint_path = Path(checkpoint)
        
        # Check if already in safetensors format
        existing_safetensors = list(checkpoint_path.glob("*.safetensors")) if checkpoint_path.is_dir() else []
        
        if existing_safetensors:
            # Copy existing safetensors
            try:
                shutil.copy(existing_safetensors[0], safetensors_path)
                self.logger.info("Copied existing safetensors: %s", safetensors_path)
                return safetensors_path
            except Exception as exc:
                self.logger.warning("Could not copy safetensors: %s", exc)
                return None
        
        # Convert PyTorch checkpoint to safetensors
        try:
            import torch
            from safetensors.torch import save_file
            
            # Find and load checkpoint
            if checkpoint_path.is_dir():
                pt_files = list(checkpoint_path.glob("*.pt"))
                if not pt_files:
                    return None
                state_dict = torch.load(pt_files[0], map_location="cpu")
            else:
                state_dict = torch.load(checkpoint_path, map_location="cpu")
            
            # Extract model state dict if wrapped
            if isinstance(state_dict, dict):
                if "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
                elif "model" in state_dict:
                    state_dict = state_dict["model"]
            
            # Save as safetensors
            save_file(state_dict, str(safetensors_path))
            self.logger.info("Converted to safetensors: %s", safetensors_path)
            return safetensors_path
            
        except ImportError:
            self.logger.warning("safetensors not available for export")
            return None
        except Exception as exc:
            self.logger.warning("Safetensors export error: %s", exc)
            return None
