"""
Rust/Candle Runner for Ray Pipeline
====================================

This runner integrates with the actual Rust implementation from:

- ``Deepseek-from-scratch-in-rust/src/model/`` - Core model components
    - ``attention.rs`` - FlashAttention, MLAAttention
    - ``moe.rs`` - MoE, DeepSeekMoE expert routing
    - ``mtp.rs`` - Multi-Token Prediction
    - ``mla.rs`` - Multi-Head Latent Attention
    - ``quantization.rs`` - FP8 quantization
    
- ``Deepseek-from-scratch-in-rust/src/training/`` - Training infrastructure
    - ``training.rs`` - TrainingConfig, training loop
    - ``grpo.rs`` - GRPO implementation
    - ``sft.rs`` - SFTConfig, SFTTrainer
    - ``distillation.rs`` - Knowledge distillation
    
- ``Deepseek-from-scratch-in-rust/src/distributed/`` - Distribution
    - ``parallel.rs`` - TensorParallel, PipelineParallel
    - ``ring_attention.rs`` - Ring attention for long sequences

Usage
-----
The runner is invoked by pipeline stages when backend is RUST::

    from deepseek.pipeline.runners import RustRunner
    from deepseek.pipeline.config import PipelineConfig, Backend

    config = PipelineConfig.from_size(ModelSize.SMALL)
    config.backend = Backend.RUST
    runner = RustRunner(config, stage="pretrain")
    result = runner.run()

Backend Requirements
--------------------
- Rust toolchain installed
- Cargo build environment
- CUDA toolkit (optional, for GPU support)

Implementation Details
----------------------
This runner executes the Rust binary via subprocess with JSON-formatted
configuration. The Rust implementation uses Candle for tensor operations
and supports:

- FP8 mixed precision training
- Tensor parallelism
- Pipeline parallelism
- Ring attention
- Gradient checkpointing

Supported Cargo Commands
------------------------
- ``cargo run --release -- pretrain`` - Pretraining
- ``cargo run --release -- sft`` - Supervised fine-tuning
- ``cargo run --release -- grpo`` - GRPO alignment
- ``cargo run --release -- distill`` - Knowledge distillation
- ``cargo run --release -- benchmark`` - Performance benchmarks
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from deepseek.pipeline.runners.base import BaseRunner, RunnerResult
from deepseek.pipeline.config import PipelineConfig


# Stage name mapping: Ray pipeline stage -> Rust CLI subcommand
RUST_STAGE_COMMANDS = {
    "pretrain": "pretrain",
    "sft": "sft",
    "grpo": "grpo",
    "distillation": "distill",
    "export": "export",
    "benchmark": "benchmark",
}


class RustRunner(BaseRunner):
    """
    Runs Rust/Candle implementation for high-performance training.
    
    This runner builds and executes the Rust binary from 
    ``Deepseek-from-scratch-in-rust/`` with appropriate configuration.
    
    The Rust implementation uses Candle (Hugging Face's Rust ML framework)
    and provides:
    
    - FP8 quantization (``src/model/quantization.rs``)
    - MLAAttention (``src/model/mla.rs``)
    - DeepSeekMoE (``src/model/moe.rs``)
    - Multi-Token Prediction (``src/model/mtp.rs``)
    - GRPO training (``src/training/grpo.rs``)
    - Distributed training (``src/distributed/``)
    
    Attributes
    ----------
    name : str
        Runner identifier ("rust")
    stage : str
        Current pipeline stage
    crate_dir : Path
        Path to Rust crate
        
    Examples
    --------
    >>> config = PipelineConfig.from_size(ModelSize.SMALL)
    >>> config.backend = Backend.RUST
    >>> runner = RustRunner(config, stage="pretrain")
    >>> result = runner.run(dataset_uri="./cache/train")
    """

    name = "rust"

    def __init__(self, config: PipelineConfig, stage: str):
        """
        Initialize the Rust runner.
        
        Parameters
        ----------
        config : PipelineConfig
            Pipeline configuration
        stage : str
            Pipeline stage name
        """
        super().__init__(config)
        self.stage = stage
        self.repo_root = Path(__file__).resolve().parents[4]  # Go up to DeepSeek-From-Scratch
        self.crate_dir = self.repo_root / "rust-src"

    def _build_rust_config(
        self,
        dataset_uri: Optional[str] = None,
        pad_token_id: int = 0,
        training_config: Optional[Dict[str, Any]] = None,
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build configuration JSON for Rust binary.
        
        This maps PipelineConfig fields to Rust's TrainingConfig and ModelConfig
        structures as defined in:
        
        - ``src/utils/config.rs`` - DeepSeekConfig
        - ``src/training/training.rs`` - TrainingConfig
        
        Parameters
        ----------
        dataset_uri : str, optional
            Path to dataset
        pad_token_id : int
            Padding token ID
        training_config : Dict, optional
            Training overrides
        extra_config : Dict, optional
            Additional configuration
            
        Returns
        -------
        Dict
            JSON-serializable config for Rust binary
        """
        model_cfg = training_config or {}
        
        rust_config = {
            # Model configuration (maps to src/utils/config.rs DeepSeekConfig)
            "model": {
                "d_model": model_cfg.get("d_model", self.config.model.d_model),
                "num_layers": model_cfg.get("num_layers", self.config.model.num_layers),
                "num_heads": model_cfg.get("num_heads", self.config.model.num_heads),
                "vocab_size": model_cfg.get("vocab_size", self.config.model.vocab_size),
                "max_seq_len": self.config.model.max_seq_len,
                "d_latent": self.config.model.d_latent,
                "d_rope": self.config.model.d_rope,
                # MoE config (maps to src/model/moe.rs)
                "num_experts": self.config.model.num_experts,
                "num_shared_experts": self.config.model.num_shared_experts,
                "top_k": self.config.model.top_k,
                # MTP config (maps to src/model/mtp.rs)
                "mtp_k": self.config.model.mtp_k,
            },
            # Training config (maps to src/training/training.rs TrainingConfig)
            "training": {
                "batch_size": self.config.training.batch_size,
                "learning_rate": self.config.training.learning_rate,
                "warmup_steps": self.config.training.warmup_steps,
                "max_steps": self.config.training.max_steps,
                "weight_decay": self.config.training.weight_decay,
                "gradient_clipping": self.config.training.max_grad_norm,
                # FP8 config (maps to src/model/quantization.rs)
                "use_fp8": getattr(self.config.training, "use_fp8", False),
                "use_amp": self.config.training.use_amp,
                "gradient_checkpointing": self.config.training.gradient_checkpointing,
            },
            # Data config
            "data": {
                "dataset_uri": dataset_uri or "",
                "pad_token_id": pad_token_id,
            },
            # Distributed config
            "distributed": {
                "sequence_parallel_size": self.config.distributed.sequence_parallel_size,
                "pipeline_parallel_size": self.config.distributed.pipeline_parallel_size,
                "data_parallel_size": self.config.distributed.data_parallel_size,
                "tensor_parallel_size": self.config.distributed.tensor_parallel_size,
            },
            # Stage-specific
            "stage": self.stage,
        }
        
        # Add stage-specific config
        if self.stage == "sft":
            # Maps to src/training/sft.rs SFTConfig
            rust_config["sft"] = {
                "lora_r": self.config.sft.lora_r,
                "lora_alpha": self.config.sft.lora_alpha,
                "lora_dropout": self.config.sft.lora_dropout,
            }
        elif self.stage == "grpo":
            # Maps to src/training/grpo.rs GRPOConfig
            rust_config["grpo"] = {
                "beta": self.config.grpo.beta,
                "group_size": self.config.grpo.group_size,
            }
        elif self.stage == "distillation":
            # Maps to src/training/distillation.rs
            rust_config["distillation"] = {
                "temperature": self.config.distillation.temperature,
                "alpha": self.config.distillation.alpha,
            }
            
        if extra_config:
            rust_config.update(extra_config)
            
        return rust_config

    def run(
        self,
        dataset_uri: Optional[str] = None,
        pad_token_id: int = 0,
        training_config: Optional[Dict[str, Any]] = None,
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> RunnerResult:
        """
        Execute Rust training via cargo.
        
        This method:
        
        1. Validates the Rust crate exists
        2. Builds configuration JSON
        3. Writes config to temp file
        4. Executes ``cargo run --release`` with appropriate subcommand
        5. Parses output and returns results
        
        The Rust implementation entry points are:
        
        - ``src/main.rs`` - CLI entry point
        - ``src/training/training.rs`` - Main training loop
        
        Parameters
        ----------
        dataset_uri : str, optional
            Path to dataset directory
        pad_token_id : int
            Padding token ID
        training_config : Dict, optional
            Training hyperparameter overrides
        extra_config : Dict, optional
            Stage-specific configuration
            
        Returns
        -------
        RunnerResult
            Training metrics and artifacts
        """
        if not self.crate_dir.exists():
            raise FileNotFoundError(
                f"Rust crate not found at {self.crate_dir}. Ensure repository is complete."
            )

        cargo_toml = self.crate_dir / "Cargo.toml"
        if not cargo_toml.exists():
            raise FileNotFoundError(
                f"Cargo.toml not found at {cargo_toml}. Invalid Rust crate."
            )

        # Build configuration
        rust_config = self._build_rust_config(
            dataset_uri=dataset_uri,
            pad_token_id=pad_token_id,
            training_config=training_config,
            extra_config=extra_config,
        )
        
        # Write config to temp file
        config_path = self.crate_dir / "target" / "ray_pipeline_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(rust_config, f, indent=2)
        
        # Determine Rust subcommand
        rust_cmd = RUST_STAGE_COMMANDS.get(self.stage, "run")
        
        # Detect platform and set appropriate features
        # On macOS without CUDA, don't use cuda feature (uses Metal by default via accelerate)
        import platform
        import shutil
        
        default_features = []
        if platform.system() == "Darwin":
            # macOS - use Metal (no cuda feature needed, metal+accelerate are default in Cargo.toml)
            default_features = []
        elif shutil.which("nvidia-smi"):
            # Has NVIDIA GPU - use CUDA
            default_features = ["cuda"]
        
        features = extra_config.get("features", default_features) if extra_config else default_features
        
        # Build cargo command
        cmd = ["cargo", "run", "--release"]
        if features:
            feature_str = ",".join(features)
            cmd.extend(["--features", feature_str])
        cmd.extend([
            "--",
            rust_cmd,
            "--config", str(config_path),
        ])
        
        self.logger.info("Running Rust command: %s", " ".join(cmd))
        self.logger.info("Config file: %s", config_path)

        env = os.environ.copy()
        env["RUST_LOG"] = "info"  # Enable Rust logging
        
        # Set CUDA_VISIBLE_DEVICES for PP stages
        if extra_config and "gpu_ids" in extra_config:
            gpu_ids = extra_config["gpu_ids"]
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
            self.logger.info("CUDA_VISIBLE_DEVICES=%s", env["CUDA_VISIBLE_DEVICES"])
        
        # Check if this is a dry run (don't actually execute)
        dry_run = extra_config.get("dry_run", False) if extra_config else False

        if dry_run:
            self.logger.info("Dry run mode - not executing Rust binary")
            return RunnerResult(
                metrics={
                    "dry_run": True,
                    "stage": self.stage,
                    "backend": "rust",
                    "config": rust_config,
                },
                checkpoint_path=None,
                artifacts={"config_path": str(config_path)},
                extra={"stage": self.stage},
            )

        # Configuration for execution
        timeout_seconds = extra_config.get("timeout", 3600 * 24) if extra_config else 3600 * 24  # 24h default
        max_retries = extra_config.get("max_retries", 2) if extra_config else 2
        stream_output = extra_config.get("stream_output", True) if extra_config else True

        # Execute with retry logic
        for attempt in range(max_retries + 1):
            if attempt > 0:
                self.logger.warning("Retry attempt %d/%d for Rust training", attempt, max_retries)

            if stream_output:
                # Stream output in real-time for long-running jobs
                metrics, artifacts = self._run_with_streaming(cmd, env, timeout_seconds)
            else:
                # Capture all output at once
                metrics, artifacts = self._run_captured(cmd, env, timeout_seconds)

            # Check if successful
            if metrics.get("returncode", -1) == 0:
                break

            # Check if retryable error (OOM, timeout, etc.)
            error_msg = metrics.get("error_message", "")
            retryable = any(x in error_msg.lower() for x in ["out of memory", "timeout", "resource"])
            if not retryable or attempt >= max_retries:
                break

            self.logger.info("Retryable error detected, waiting before retry...")
            import time
            time.sleep(30)  # Wait 30s before retry

        return RunnerResult(
            metrics=metrics,
            checkpoint_path=None,
            artifacts=artifacts,
            extra={"stage": self.stage, "config": rust_config},
        )

    def _run_with_streaming(
        self,
        cmd: list,
        env: dict,
        timeout_seconds: int,
    ) -> tuple:
        """Run command with streaming output for real-time monitoring."""
        metrics = {
            "returncode": -1,
            "stage": self.stage,
            "backend": "rust",
        }
        stdout_lines = []
        stderr_lines = []

        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(self.crate_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )

            import select
            import time

            start_time = time.time()

            while True:
                # Check timeout
                if time.time() - start_time > timeout_seconds:
                    process.terminate()
                    process.wait(timeout=10)
                    metrics["error"] = True
                    metrics["error_message"] = f"Timeout after {timeout_seconds}s"
                    break

                # Use select to check for output (Unix-only, falls back on Windows)
                try:
                    readable, _, _ = select.select([process.stdout, process.stderr], [], [], 1.0)
                except (ValueError, OSError):
                    # Fallback for Windows or closed pipes
                    readable = []

                for stream in readable:
                    line = stream.readline()
                    if line:
                        if stream == process.stdout:
                            stdout_lines.append(line.strip())
                            # Parse metrics from JSON output
                            if line.strip().startswith('{"metrics":'):
                                try:
                                    parsed = json.loads(line.strip())
                                    metrics.update(parsed.get("metrics", {}))
                                except json.JSONDecodeError:
                                    pass
                            # Log progress lines
                            if "step" in line.lower() or "loss" in line.lower():
                                self.logger.info("[Rust] %s", line.strip())
                        else:
                            stderr_lines.append(line.strip())

                # Check if process has finished
                retcode = process.poll()
                if retcode is not None:
                    # Read any remaining output
                    remaining_stdout, remaining_stderr = process.communicate()
                    if remaining_stdout:
                        stdout_lines.extend(remaining_stdout.strip().split('\n'))
                    if remaining_stderr:
                        stderr_lines.extend(remaining_stderr.strip().split('\n'))
                    metrics["returncode"] = retcode
                    break

        except Exception as e:
            metrics["error"] = True
            metrics["error_message"] = str(e)
            self.logger.exception("Error running Rust command")

        if metrics.get("returncode", -1) != 0:
            metrics["error"] = True
            metrics["error_message"] = "\n".join(stderr_lines[-10:])
            self.logger.error("Rust training failed: %s", metrics["error_message"])

        artifacts = {
            "stdout": "\n".join(stdout_lines),
            "stderr": "\n".join(stderr_lines),
            "config_path": str(self.crate_dir / "target" / "ray_pipeline_config.json"),
        }

        return metrics, artifacts

    def _run_captured(
        self,
        cmd: list,
        env: dict,
        timeout_seconds: int,
    ) -> tuple:
        """Run command with captured output (simpler but no real-time streaming)."""
        metrics = {
            "returncode": -1,
            "stage": self.stage,
            "backend": "rust",
        }

        try:
            process = subprocess.run(
                cmd,
                cwd=str(self.crate_dir),
                capture_output=True,
                text=True,
                check=False,
                env=env,
                timeout=timeout_seconds,
            )

            metrics["returncode"] = process.returncode

            # Try to parse metrics from stdout
            for line in process.stdout.split("\n"):
                if line.startswith('{"metrics":'):
                    try:
                        parsed = json.loads(line)
                        metrics.update(parsed.get("metrics", {}))
                        break
                    except json.JSONDecodeError:
                        pass

            if process.returncode != 0:
                metrics["error"] = True
                metrics["error_message"] = process.stderr[-500:]
                self.logger.error("Rust training failed: %s", process.stderr[-1000:])

            artifacts = {
                "stdout": process.stdout,
                "stderr": process.stderr,
                "config_path": str(self.crate_dir / "target" / "ray_pipeline_config.json"),
            }

        except subprocess.TimeoutExpired:
            metrics["error"] = True
            metrics["error_message"] = f"Timeout after {timeout_seconds}s"
            artifacts = {"stdout": "", "stderr": "", "config_path": ""}
        except Exception as e:
            metrics["error"] = True
            metrics["error_message"] = str(e)
            artifacts = {"stdout": "", "stderr": "", "config_path": ""}

        return metrics, artifacts
