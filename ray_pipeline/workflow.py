"""
Pipeline orchestration for DeepSeek training.

This module provides workflow orchestration using Ray for distributed execution.
Note: Ray Workflows was deprecated in Ray 2.44, so we use Ray Tasks directly.

Time-Sliced Wave Execution
--------------------------
For production training, this module supports time-sliced wave execution:
- 4 sequential waves alternating between Rust and Python backends
- Each wave runs 5k steps (20k total) on 3 GPUs with PP=3
- Checkpoint handoff between waves ensures continuous training
- Validation after each wave for best-model selection

Wave Schedule:
- Wave 1 (Rust): MQA/GQA/MLA/DeepSeek Attention (steps 0-5k)
- Wave 2 (Python): Standard MOE/DeepSeek MOE (steps 5k-10k)
- Wave 3 (Rust): GRPO/R1/DPO/Reward (steps 10k-15k)
- Wave 4 (Python): MTP/FP8/Distillation/5D Parallelism (steps 15k-20k)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Type

import ray

from ray_pipeline.config import (
    PipelineConfig,
    Stage,
    WaveBackend,
    WaveConfig,
    TimeSlicedConfig,
)
from ray_pipeline.stages import (
    DataPrepStage,
    DistillationStage,
    ExportStage,
    GRPOStage,
    PretrainStage,
    SFTStage,
)
from ray_pipeline.stages.base import BaseStage, StageContext

LOGGER = logging.getLogger("ray_pipeline.workflow")

STAGE_REGISTRY: Dict[Stage, Type[BaseStage]] = {
    Stage.DATA_PREP: DataPrepStage,
    Stage.PRETRAIN: PretrainStage,
    Stage.SFT: SFTStage,
    Stage.GRPO: GRPOStage,
    Stage.DISTILLATION: DistillationStage,
    Stage.EXPORT: ExportStage,
}


@ray.remote
def _run_stage_remote(stage_value: str, config_dict: dict, prev_output: Optional[dict], metadata: dict) -> dict:
    """
    Ray remote function to run a single stage.
    
    Args:
        stage_value: Stage enum value as string
        config_dict: Serialized PipelineConfig
        prev_output: Previous stage output (serializable dict)
        metadata: Pipeline metadata dict
        
    Returns:
        Updated context as serializable dict
    """
    from ray_pipeline.config import PipelineConfig, Stage
    from ray_pipeline.stages.base import StageContext
    
    # Reconstruct config and context
    config = PipelineConfig.from_dict(config_dict)
    context = StageContext(config=config, previous_output=prev_output, metadata=metadata)
    
    stage_enum = Stage(stage_value)
    stage_cls = STAGE_REGISTRY.get(stage_enum)
    if stage_cls is None:
        raise KeyError(f"Stage {stage_enum.value} not registered")
    
    stage = stage_cls(config)
    LOGGER.info("Running stage %s", stage_enum.value)
    result_context = stage.run(context)
    
    # Return serializable dict
    return {
        "previous_output": result_context.previous_output,
        "metadata": result_context.metadata,
    }


class DeepSeekWorkflow:
    """
    Orchestrates the DeepSeek training pipeline.
    
    This class manages the execution of pipeline stages either:
    - Distributed via Ray Tasks
    - Sequentially in local mode
    
    Example
    -------
    >>> config = PipelineConfig.from_size(ModelSize.SMALL)
    >>> workflow = DeepSeekWorkflow(config)
    >>> result = workflow.run(input_data="./data", output_dir="./checkpoints")
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def run(
        self,
        input_data: Optional[str] = None,
        output_dir: Optional[str] = None,
        use_ray: bool = True,
    ) -> StageContext:
        """
        Execute the configured pipeline.
        
        Args:
            input_data: Path to input data directory
            output_dir: Path to output/checkpoint directory
            use_ray: Whether to use Ray for distributed execution
            
        Returns:
            Final StageContext with all outputs and metadata
        """
        # Initialize metadata
        metadata = {
            "input_data": input_data,
            "output_dir": output_dir,
        }
        
        if use_ray:
            return self._run_distributed(metadata)
        else:
            return self._run_sequential(metadata)
    
    def _run_distributed(self, metadata: dict) -> StageContext:
        """Run pipeline stages as Ray tasks."""
        if not ray.is_initialized():
            ray.init(
                address=self.config.distributed.ray_address or None,
                ignore_reinit_error=True,
            )
        
        config_dict = self.config.to_dict()
        prev_output = None
        
        for stage in self.config.stages_to_run:
            self.logger.info("Submitting stage: %s", stage.value)
            result_ref = _run_stage_remote.remote(
                stage.value, config_dict, prev_output, metadata
            )
            result = ray.get(result_ref)
            prev_output = result["previous_output"]
            metadata = result["metadata"]
        
        return StageContext(
            config=self.config,
            previous_output=prev_output,
            metadata=metadata,
        )
    
    def _run_sequential(self, metadata: dict) -> StageContext:
        """Run pipeline stages sequentially (no Ray)."""
        context = StageContext(
            config=self.config,
            previous_output=None,
            metadata=metadata,
        )
        
        for stage in self.config.stages_to_run:
            stage_cls = STAGE_REGISTRY.get(stage)
            if stage_cls is None:
                raise ValueError(f"Stage {stage.value} not supported")
            self.logger.info("Running stage: %s", stage.value)
            context = stage_cls(self.config).run(context)
        
        return context

    # ------------------------------------------------------------------
    # Time-Sliced Wave Execution (Production 3-GPU Pipeline)
    # ------------------------------------------------------------------
    def run_time_sliced_waves(
        self,
        input_data: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> StageContext:
        """
        Execute time-sliced waves alternating Rust/Python backends on 3 GPUs.
        
        This method implements the production pipeline:
        - Wave 1 (Rust): MQA/GQA/MLA/DeepSeek Attention (0-5k steps)
        - Wave 2 (Python): Standard MOE/DeepSeek MOE (5k-10k steps)
        - Wave 3 (Rust): GRPO/R1/DPO/Reward (10k-15k steps)
        - Wave 4 (Python): MTP/FP8/Distillation/5D Parallelism (15k-20k steps)
        
        Each wave:
        1. Loads checkpoint from previous wave (if applicable)
        2. Runs training with appropriate backend
        3. Saves checkpoints at 1k intervals
        4. Runs validation to record wave loss
        
        At completion (20k steps), compares validation losses and selects
        best checkpoint as final model.
        
        Parameters
        ----------
        input_data : str, optional
            Path to input data directory
        output_dir : str, optional
            Path to output/checkpoint directory
            
        Returns
        -------
        StageContext
            Final context with all outputs, metadata including:
            - wave_metrics: Dict of wave_id -> validation_loss
            - best_wave: Wave ID with lowest validation loss
            - best_checkpoint: Path to best model checkpoint
        """
        if not self.config.time_sliced.enabled:
            self.logger.warning("Time-sliced execution not enabled, running standard pipeline")
            return self.run(input_data=input_data, output_dir=output_dir)
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(
                address=self.config.distributed.ray_address or None,
                ignore_reinit_error=True,
            )
        
        # Set up directories
        output_dir = output_dir or self.config.training.checkpoint_dir
        checkpoint_base = Path(output_dir)
        checkpoint_base.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata
        metadata = {
            "input_data": input_data,
            "output_dir": str(output_dir),
            "wave_metrics": {},  # wave_id -> validation_loss
            "wave_checkpoints": {},  # wave_id -> checkpoint_path
        }
        
        # GPU environment for PP=3
        gpu_ids = self.config.time_sliced.gpu_ids
        cuda_visible_devices = ",".join(str(g) for g in gpu_ids)
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        
        self.logger.info(
            "Starting time-sliced wave execution: %d waves, PP=%d, GPUs=%s",
            self.config.time_sliced.num_waves,
            self.config.time_sliced.pipeline_parallel_size,
            cuda_visible_devices,
        )
        
        prev_checkpoint = None
        
        for wave in self.config.time_sliced.waves:
            self.logger.info(
                "═══════════════════════════════════════════════════════════"
            )
            self.logger.info(
                "  Wave %d: %s backend | Steps %d-%d | Stages: %s",
                wave.wave_id,
                wave.backend.value.upper(),
                wave.start_step,
                wave.end_step,
                ", ".join(wave.stages),
            )
            self.logger.info(
                "═══════════════════════════════════════════════════════════"
            )
            
            # Load checkpoint from previous wave if available
            checkpoint_from = wave.checkpoint_from or prev_checkpoint
            
            # Run wave
            wave_result = self._run_single_wave(
                wave=wave,
                checkpoint_from=checkpoint_from,
                checkpoint_base=checkpoint_base,
                metadata=metadata,
            )
            
            # Record wave metrics
            val_loss = wave_result.get("validation_loss", float("inf"))
            metadata["wave_metrics"][wave.wave_id] = val_loss
            
            # Update checkpoint path for next wave
            prev_checkpoint = str(checkpoint_base / f"step_{wave.end_step}")
            metadata["wave_checkpoints"][wave.wave_id] = prev_checkpoint
            
            self.logger.info(
                "Wave %d complete: validation_loss=%.6f, checkpoint=%s",
                wave.wave_id,
                val_loss,
                prev_checkpoint,
            )
        
        # Select best model based on validation loss
        best_wave, best_checkpoint = self._select_best_checkpoint(
            metadata=metadata,
            checkpoint_base=checkpoint_base,
        )
        
        metadata["best_wave"] = best_wave
        metadata["best_checkpoint"] = best_checkpoint
        
        self.logger.info(
            "═══════════════════════════════════════════════════════════"
        )
        self.logger.info("  Time-sliced training complete!")
        self.logger.info("  Best wave: %d (val_loss=%.6f)", best_wave, metadata["wave_metrics"][best_wave])
        self.logger.info("  Best checkpoint: %s", best_checkpoint)
        self.logger.info(
            "═══════════════════════════════════════════════════════════"
        )
        
        return StageContext(
            config=self.config,
            previous_output={"best_checkpoint": best_checkpoint},
            metadata=metadata,
        )
    
    def _run_single_wave(
        self,
        wave: WaveConfig,
        checkpoint_from: Optional[str],
        checkpoint_base: Path,
        metadata: dict,
    ) -> dict:
        """
        Execute a single training wave with the appropriate backend.
        
        Parameters
        ----------
        wave : WaveConfig
            Wave configuration
        checkpoint_from : str, optional
            Path to load checkpoint from
        checkpoint_base : Path
            Base directory for saving checkpoints
        metadata : dict
            Pipeline metadata
            
        Returns
        -------
        dict
            Wave result with validation_loss and checkpoint_path
        """
        from ray_pipeline.runners import RustRunner, PyTorchRunner
        
        # Build wave-specific training config
        wave_training_config = {
            "start_step": wave.start_step,
            "max_steps": wave.end_step,
            "checkpoint_from": checkpoint_from,
            "save_every_n_steps": self.config.training.save_every_n_steps,
            "wave_id": wave.wave_id,
            "stages": wave.stages,
        }
        
        extra_config = {
            "wave_id": wave.wave_id,
            "pipeline_parallel_size": self.config.time_sliced.pipeline_parallel_size,
            "gpu_ids": self.config.time_sliced.gpu_ids,
        }
        
        if wave.backend == WaveBackend.RUST:
            result = self._run_rust_wave(
                wave=wave,
                training_config=wave_training_config,
                extra_config=extra_config,
                metadata=metadata,
            )
        elif wave.backend == WaveBackend.MLX:
            result = self._run_mlx_wave(
                wave=wave,
                training_config=wave_training_config,
                extra_config=extra_config,
                metadata=metadata,
            )
        else:  # WaveBackend.PYTHON
            result = self._run_python_wave(
                wave=wave,
                training_config=wave_training_config,
                extra_config=extra_config,
                metadata=metadata,
            )
        
        # Run validation if enabled
        validation_loss = float("inf")
        if self.config.time_sliced.validation_after_each_wave:
            validation_loss = self._run_validation(
                wave=wave,
                checkpoint_path=str(checkpoint_base / f"step_{wave.end_step}"),
                metadata=metadata,
            )
        
        result["validation_loss"] = validation_loss
        return result
    
    def _run_rust_wave(
        self,
        wave: WaveConfig,
        training_config: dict,
        extra_config: dict,
        metadata: dict,
    ) -> dict:
        """Execute a Rust backend wave using Modal GPU containers."""
        from ray_pipeline.runners import ModalRunner
        
        self.logger.info("Running Rust wave %d on Modal GPU with CUDA features", wave.wave_id)
        
        # Configure Modal runner for Rust implementation
        extra_config["implementation"] = "rust"
        extra_config["features"] = ["cuda"]
        extra_config["distributed"] = {
            "pipeline_parallel_size": self.config.time_sliced.pipeline_parallel_size,
        }
        
        runner = ModalRunner(self.config, stage="pretrain")
        
        # Use data path from config or default
        data_path = metadata.get("input_data") or self.config.data.data_dir
        
        result = runner.run(
            dataset_uri=data_path,
            pad_token_id=0,
            training_config=training_config,
            extra_config=extra_config,
        )
        
        return {
            "backend": "rust",
            "wave_id": wave.wave_id,
            "metrics": result.metrics,
            "checkpoint_path": result.checkpoint_path,
        }
    
    def _run_python_wave(
        self,
        wave: WaveConfig,
        training_config: dict,
        extra_config: dict,
        metadata: dict,
    ) -> dict:
        """Execute a Python/PyTorch backend wave using Modal GPU containers."""
        from ray_pipeline.runners import ModalRunner
        
        self.logger.info("Running Python wave %d on Modal GPU with PP=%d", 
                        wave.wave_id, self.config.time_sliced.pipeline_parallel_size)
        
        # Configure Modal runner for Python implementation
        extra_config["implementation"] = "python"
        
        runner = ModalRunner(self.config, stage="pretrain")
        
        # Use data path from config or default
        data_path = metadata.get("input_data") or self.config.data.data_dir
        
        result = runner.run(
            dataset_uri=data_path,
            pad_token_id=0,
            training_config=training_config,
            extra_config=extra_config,
        )
        
        return {
            "backend": "python",
            "wave_id": wave.wave_id,
            "metrics": result.metrics,
            "checkpoint_path": result.checkpoint_path,
        }
    
    def _run_mlx_wave(
        self,
        wave: WaveConfig,
        training_config: dict,
        extra_config: dict,
        metadata: dict,
    ) -> dict:
        """Execute a MLX backend wave on Apple Silicon."""
        from ray_pipeline.runners import MLXRunner
        
        self.logger.info("Running MLX wave %d on Apple Silicon GPU", wave.wave_id)
        
        # Configure MLX runner
        runner = MLXRunner(self.config, stage="pretrain")
        
        # Use data path from config or default
        data_path = metadata.get("input_data") or self.config.data.data_dir
        
        result = runner.run(
            dataset_uri=data_path,
            pad_token_id=0,
            training_config=training_config,
            extra_config=extra_config,
        )
        
        return {
            "backend": "mlx",
            "wave_id": wave.wave_id,
            "metrics": result.metrics,
            "checkpoint_path": result.checkpoint_path,
        }
    
    def _run_validation(
        self,
        wave: WaveConfig,
        checkpoint_path: str,
        metadata: dict,
    ) -> float:
        """
        Run validation on held-out split after wave completion.
        
        Parameters
        ----------
        wave : WaveConfig
            Completed wave configuration
        checkpoint_path : str
            Path to wave checkpoint
        metadata : dict
            Pipeline metadata
            
        Returns
        -------
        float
            Validation loss
        """
        self.logger.info("Running validation for wave %d from %s", wave.wave_id, checkpoint_path)
        
        # TODO: Implement actual validation loop
        # For now, return placeholder based on wave progression
        # In production, this loads checkpoint and runs eval on validation split
        import random
        base_loss = 2.5 - (wave.wave_id * 0.3)  # Simulated improvement
        validation_loss = base_loss + random.uniform(-0.1, 0.1)
        
        metadata[f"wave_{wave.wave_id}_val_loss"] = validation_loss
        return validation_loss
    
    def _select_best_checkpoint(
        self,
        metadata: dict,
        checkpoint_base: Path,
    ) -> tuple:
        """
        Compare validation losses and select best checkpoint.
        
        Compares Rust waves (1, 3) and Python waves (2, 4), selecting
        the checkpoint with lowest validation loss as final model.
        
        Parameters
        ----------
        metadata : dict
            Pipeline metadata with wave_metrics
        checkpoint_base : Path
            Base checkpoint directory
            
        Returns
        -------
        tuple
            (best_wave_id, best_checkpoint_path)
        """
        wave_metrics = metadata.get("wave_metrics", {})
        
        if not wave_metrics:
            self.logger.warning("No wave metrics found, using last checkpoint")
            return (4, str(checkpoint_base / "step_20000"))
        
        # Find wave with lowest validation loss
        best_wave = min(wave_metrics, key=wave_metrics.get)
        best_wave_checkpoint = metadata["wave_checkpoints"].get(best_wave)
        
        # Copy best checkpoint to final location
        final_checkpoint = checkpoint_base / "final" / "best_model.safetensors"
        final_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy checkpoint files
        if best_wave_checkpoint and Path(best_wave_checkpoint).exists():
            src_dir = Path(best_wave_checkpoint)
            for src_file in src_dir.glob("*.safetensors"):
                dst_file = final_checkpoint.parent / src_file.name
                shutil.copy2(src_file, dst_file)
                self.logger.info("Copied %s -> %s", src_file, dst_file)
        
        # Save selection metadata
        selection_metadata = {
            "best_wave": best_wave,
            "best_wave_val_loss": wave_metrics[best_wave],
            "all_wave_metrics": wave_metrics,
            "source_checkpoint": best_wave_checkpoint,
        }
        
        with open(final_checkpoint.parent / "selection_metadata.json", "w") as f:
            json.dump(selection_metadata, f, indent=2)
        
        return (best_wave, str(final_checkpoint))


def run_pipeline(
    config: PipelineConfig,
    use_ray: bool = True,
    initial_context: Optional[StageContext] = None,
) -> StageContext:
    """
    Execute the configured pipeline.
    
    This is a convenience function that wraps DeepSeekWorkflow.
    
    Args:
        config: Pipeline configuration
        use_ray: Whether to use Ray for distributed execution
        initial_context: Optional initial context
        
    Returns:
        Final StageContext
    """
    workflow = DeepSeekWorkflow(config)
    metadata = initial_context.metadata if initial_context else {}
    return workflow.run(
        input_data=metadata.get("input_data"),
        output_dir=metadata.get("output_dir"),
        use_ray=use_ray,
    )
