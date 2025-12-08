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

from deepseek.pipeline.config import (
    PipelineConfig,
    Stage,
    WaveBackend,
    WaveConfig,
    TimeSlicedConfig,
)
from deepseek.pipeline.stages import (
    DataPrepStage,
    DistillationStage,
    ExportStage,
    GRPOStage,
    PretrainStage,
    SFTStage,
)
from deepseek.pipeline.stages.base import BaseStage, StageContext

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
    from deepseek.pipeline.config import PipelineConfig, Stage
    from deepseek.pipeline.stages.base import StageContext
    
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
            import sys
            ray.init(
                address=self.config.distributed.ray_address or None,
                ignore_reinit_error=True,
                runtime_env={
                    "env_vars": {"PYTHONPATH": ":".join(sys.path)},
                },
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
            import sys
            ray.init(
                address=self.config.distributed.ray_address or None,
                ignore_reinit_error=True,
                runtime_env={
                    "env_vars": {"PYTHONPATH": ":".join(sys.path)},
                },
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
        from deepseek.pipeline.runners import RustRunner, PyTorchRunner
        
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
        elif wave.backend in (WaveBackend.RUST_METAL, WaveBackend.RUST_CUDA, WaveBackend.RUST_CPU):
            # Handle Rust backend variants - all use the same runner with different device config
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
        elif wave.backend == WaveBackend.PYTORCH_CUDA:
            result = self._run_pytorch_cuda_wave(
                wave=wave,
                training_config=wave_training_config,
                extra_config=extra_config,
                metadata=metadata,
            )
        elif wave.backend == WaveBackend.PYTORCH_MPS:
            result = self._run_pytorch_mps_wave(
                wave=wave,
                training_config=wave_training_config,
                extra_config=extra_config,
                metadata=metadata,
            )
        elif wave.backend == WaveBackend.CPU:
            result = self._run_cpu_wave(
                wave=wave,
                training_config=wave_training_config,
                extra_config=extra_config,
                metadata=metadata,
            )
        else:
            raise ValueError(f"Unknown backend: {wave.backend}")
        
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
    
    def _run_cpu_wave(
        self,
        wave: WaveConfig,
        training_config: dict,
        extra_config: dict,
        metadata: dict,
    ) -> dict:
        """Execute a CPU-bound wave (pure Python, no GPU)."""
        from deepseek.pipeline.runners import PyTorchRunner
        
        self.logger.info("Executing CPU wave %d with stages: %s", wave.wave_id, wave.stages)
        
        # Force CPU device
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        # Use first stage or default to pretrain
        stage = wave.stages[0] if wave.stages else "pretrain"
        runner = PyTorchRunner(self.config, stage=stage)
        runner.device = "cpu"
        
        result = runner.run(
            stages=wave.stages,
            start_step=training_config["start_step"],
            max_steps=training_config["max_steps"],
            checkpoint_from=training_config.get("checkpoint_from"),
            **extra_config,
        )
        
        return {
            "checkpoint_path": result.checkpoint_path,
            "metrics": result.metrics,
        }
    
    def _run_pytorch_cuda_wave(
        self,
        wave: WaveConfig,
        training_config: dict,
        extra_config: dict,
        metadata: dict,
    ) -> dict:
        """Execute a PyTorch+CUDA wave on GPU."""
        from deepseek.pipeline.runners import PyTorchRunner
        
        self.logger.info("Executing PyTorch+CUDA wave %d with stages: %s", wave.wave_id, wave.stages)
        
        # Use first stage or default to pretrain
        stage = wave.stages[0] if wave.stages else "pretrain"
        runner = PyTorchRunner(self.config, stage=stage)
        runner.device = "cuda"
        
        result = runner.run(
            stages=wave.stages,
            start_step=training_config["start_step"],
            max_steps=training_config["max_steps"],
            checkpoint_from=training_config.get("checkpoint_from"),
            **extra_config,
        )
        
        return {
            "checkpoint_path": result.checkpoint_path,
            "metrics": result.metrics,
        }
    
    def _run_pytorch_mps_wave(
        self,
        wave: WaveConfig,
        training_config: dict,
        extra_config: dict,
        metadata: dict,
    ) -> dict:
        """Execute a PyTorch+MPS wave on Apple Silicon GPU."""
        from deepseek.pipeline.runners import PyTorchRunner
        
        self.logger.info("Executing PyTorch+MPS wave %d with stages: %s", wave.wave_id, wave.stages)
        
        # Use first stage or default to pretrain
        stage = wave.stages[0] if wave.stages else "pretrain"
        runner = PyTorchRunner(self.config, stage=stage)
        runner.device = "mps"
        
        result = runner.run(
            stages=wave.stages,
            start_step=training_config["start_step"],
            max_steps=training_config["max_steps"],
            checkpoint_from=training_config.get("checkpoint_from"),
            **extra_config,
        )
        
        return {
            "checkpoint_path": result.checkpoint_path,
            "metrics": result.metrics,
        }

    def _run_rust_wave(
        self,
        wave: WaveConfig,
        training_config: dict,
        extra_config: dict,
        metadata: dict,
    ) -> dict:
        """Execute a Rust backend wave using local Rust/Candle runner."""
        from deepseek.pipeline.runners import RustRunner
        
        self.logger.info("Running Rust wave %d on local Metal/CUDA", wave.wave_id)
        
        # Configure Rust runner
        extra_config["implementation"] = "rust"
        extra_config["distributed"] = {
            "pipeline_parallel_size": self.config.time_sliced.pipeline_parallel_size,
        }
        
        runner = RustRunner(self.config, stage="pretrain")
        
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
        from deepseek.pipeline.runners import ModalRunner
        
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
        from deepseek.pipeline.runners import MLXRunner
        
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
        
        # Progressive validation thresholds per wave
        wave_thresholds = {
            1: 8.0,   # Wave 1: loss < 8.0
            2: 5.0,   # Wave 2: loss < 5.0
            3: 3.5,   # Wave 3: loss < 3.5
            4: 2.5,   # Wave 4: loss < 2.5
        }
        
        checkpoint_dir = Path(checkpoint_path)
        validation_loss = float("inf")
        
        # If specified checkpoint doesn't exist, try to find an available one
        if not checkpoint_dir.exists():
            # Try common checkpoint names
            base_dir = checkpoint_dir.parent
            candidates = ["final", "best"]
            # Also try step directories
            if base_dir.exists():
                for item in sorted(base_dir.iterdir(), reverse=True):
                    if item.is_dir() and item.name.startswith("step_"):
                        candidates.append(item.name)
            
            for candidate in candidates:
                alt_dir = base_dir / candidate
                if alt_dir.exists() and (alt_dir / "config.json").exists():
                    self.logger.info(
                        "Checkpoint %s not found, using %s instead",
                        checkpoint_path, alt_dir
                    )
                    checkpoint_dir = alt_dir
                    break
        
        try:
            # Determine backend and load appropriate model
            if wave.backend == WaveBackend.MLX:
                validation_loss = self._validate_mlx_checkpoint(checkpoint_dir, metadata)
            elif wave.backend == WaveBackend.RUST:
                validation_loss = self._validate_rust_checkpoint(checkpoint_dir, metadata)
            else:  # Python/PyTorch
                validation_loss = self._validate_pytorch_checkpoint(checkpoint_dir, metadata)
            
            # Log threshold comparison
            threshold = wave_thresholds.get(wave.wave_id, 10.0)
            if validation_loss < threshold:
                self.logger.info(
                    "Wave %d validation PASSED: loss=%.4f < threshold=%.4f",
                    wave.wave_id, validation_loss, threshold
                )
            else:
                self.logger.warning(
                    "Wave %d validation WARNING: loss=%.4f >= threshold=%.4f",
                    wave.wave_id, validation_loss, threshold
                )
                
        except Exception as e:
            self.logger.error("Validation failed for wave %d: %s", wave.wave_id, str(e))
            # Fallback: use training loss from metrics if available
            training_metrics_path = checkpoint_dir / "training_state.json"
            if training_metrics_path.exists():
                import json
                with open(training_metrics_path, "r") as f:
                    training_state = json.load(f)
                validation_loss = training_state.get("best_loss", training_state.get("last_loss", float("inf")))
                self.logger.info("Using training loss as fallback: %.4f", validation_loss)
        
        metadata[f"wave_{wave.wave_id}_val_loss"] = validation_loss
        return validation_loss
    
    def _validate_mlx_checkpoint(self, checkpoint_dir: Path, metadata: dict) -> float:
        """Validate MLX model checkpoint."""
        import sys
        from pathlib import Path as P
        
        # Add MLX implementation to path
        project_root = P(__file__).resolve().parents[1]
        try:
            import mlx.core as mx
            from deepseek.mlx.tiny_trainer import TinyMLXTrainer, TinyMTPModel, DataLoader
            from transformers import AutoTokenizer
            
            # Load checkpoint
            trainer, model, config = TinyMLXTrainer.load_checkpoint(str(checkpoint_dir))
            
            # Get validation data path
            val_data_path = metadata.get("input_data", self.config.data.data_dir)
            val_dir = Path(val_data_path)
            if (val_dir / "valid").exists():
                val_dir = val_dir / "valid"
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Create validation dataloader
            val_loader = DataLoader(
                data_path=str(val_dir),
                tokenizer=tokenizer,
                batch_size=8,
                max_seq_len=self.config.model.max_seq_len,
                shuffle=False,
            )
            
            # Run validation
            val_loss = trainer.evaluate(val_loader, tokenizer.pad_token_id, max_batches=20)
            return val_loss
            
        except ImportError as e:
            self.logger.warning("MLX validation skipped (MLX not available): %s", str(e))
            raise
    
    def _validate_pytorch_checkpoint(self, checkpoint_dir: Path, metadata: dict) -> float:
        """Validate PyTorch model checkpoint."""
        import torch
        import torch.nn.functional as F
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Check for model checkpoint
        model_path = checkpoint_dir / "model.pt"
        if not model_path.exists():
            model_path = checkpoint_dir / "checkpoint.pt"
        
        if not model_path.exists():
            raise FileNotFoundError(f"No model checkpoint found in {checkpoint_dir}")
        
        # Load model state
        checkpoint = torch.load(model_path, map_location=device)
        
        # Try to load full model config from checkpoint or use defaults
        config_path = checkpoint_dir / "config.json"
        if config_path.exists():
            import json
            with open(config_path, "r") as f:
                model_config = json.load(f)
        else:
            # Use config from pipeline
            model_config = {
                "vocab_size": self.config.model.vocab_size,
                "hidden_size": self.config.model.hidden_size,
                "num_layers": self.config.model.num_layers,
                "num_attention_heads": self.config.model.num_attention_heads,
                "intermediate_size": getattr(self.config.model, "intermediate_size", self.config.model.hidden_size * 4),
            }
        
        # Build and load model
        from deepseek.cloud.modal.distributed_trainer import _build_model
        model = _build_model(model_config).to(device)
        
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # Get validation data
        val_data_path = metadata.get("input_data", self.config.data.data_dir)
        val_dir = Path(val_data_path)
        if (val_dir / "valid").exists():
            val_dir = val_dir / "valid"
        
        # Load tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Run validation
        total_loss = 0.0
        num_batches = 0
        max_batches = 20
        
        # Simple validation loop
        import json as json_mod
        for jsonl_file in val_dir.glob("**/*.jsonl"):
            with open(jsonl_file, "r") as f:
                batch_texts = []
                for line in f:
                    try:
                        data = json_mod.loads(line)
                        if "text" in data:
                            batch_texts.append(data["text"])
                            if len(batch_texts) >= 8:
                                # Process batch
                                encoded = tokenizer(
                                    batch_texts,
                                    padding="max_length",
                                    truncation=True,
                                    max_length=self.config.training.max_seq_len,
                                    return_tensors="pt",
                                )
                                input_ids = encoded["input_ids"].to(device)
                                
                                with torch.no_grad():
                                    logits = model(input_ids)
                                    shift_logits = logits[:, :-1, :].contiguous()
                                    shift_labels = input_ids[:, 1:].contiguous()
                                    loss = F.cross_entropy(
                                        shift_logits.view(-1, model_config["vocab_size"]),
                                        shift_labels.view(-1),
                                        ignore_index=tokenizer.pad_token_id,
                                    )
                                    total_loss += loss.item()
                                    num_batches += 1
                                
                                batch_texts = []
                                if num_batches >= max_batches:
                                    break
                    except json_mod.JSONDecodeError:
                        continue
                if num_batches >= max_batches:
                    break
            if num_batches >= max_batches:
                break
        
        return total_loss / max(num_batches, 1)
    
    def _validate_rust_checkpoint(self, checkpoint_dir: Path, metadata: dict) -> float:
        """Validate Rust model checkpoint by calling Rust evaluate binary."""
        import subprocess
        import json
        
        # Get validation data path
        val_data_path = metadata.get("input_data", self.config.data.data_dir)
        
        # Write config for Rust binary
        rust_config = {
            "checkpoint_path": str(checkpoint_dir),
            "data_path": val_data_path,
            "max_batches": 20,
            "batch_size": 8,
        }
        
        config_path = checkpoint_dir / "eval_config.json"
        with open(config_path, "w") as f:
            json.dump(rust_config, f)
        
        # Find Rust binary
        project_root = Path(__file__).resolve().parents[1]
        rust_dir = project_root / "Deepseek-from-scratch-in-rust"
        
        try:
            # Try to run Rust evaluate command
            result = subprocess.run(
                ["cargo", "run", "--release", "--", "evaluate", "--config", str(config_path)],
                cwd=str(rust_dir),
                capture_output=True,
                text=True,
                timeout=300,
            )
            
            if result.returncode == 0:
                # Parse validation loss from output
                for line in result.stdout.split("\n"):
                    if "validation_loss" in line.lower():
                        try:
                            # Try to parse JSON output
                            output_data = json.loads(line)
                            return output_data.get("validation_loss", float("inf"))
                        except json.JSONDecodeError:
                            # Try to extract number
                            import re
                            match = re.search(r"[-+]?\d*\.?\d+", line)
                            if match:
                                return float(match.group())
            
            # If Rust evaluate not available, fall back to reading metrics
            self.logger.warning("Rust evaluate command not available, using training metrics")
            metrics_path = checkpoint_dir / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                return metrics.get("validation_loss", metrics.get("loss", float("inf")))
            
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            self.logger.warning("Rust validation failed: %s", str(e))
        
        # Last resort: return a reasonable default based on training metrics
        training_state_path = checkpoint_dir / "training_state.json"
        if training_state_path.exists():
            with open(training_state_path, "r") as f:
                state = json.load(f)
            return state.get("best_loss", float("inf"))
        
        return float("inf")
    
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
