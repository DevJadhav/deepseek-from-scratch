"""
Rust CUDA Runner for Modal
===========================

Runs Rust/Candle training with CUDA on Modal's A100-80GB GPUs.

This module provides:
- Rust toolchain with CUDA support
- Cargo build with --features cuda
- NCCL bindings via nccl_sys.rs
- Bi-directional checkpoint synchronization

Usage
-----
Run Rust CUDA training on Modal::

    uv run modal run modal/rust_cuda.py::train_rust

Build only (no training)::

    uv run modal run modal/rust_cuda.py::build_rust
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to path
_project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_project_root))

from modal_gpu.app import (
    RETRY_POLICY,
    app,
    checkpoints_volume,
    data_volume,
    rust_cuda_image,
    secrets,
    sync_checkpoints_to_volume,
)
from modal_gpu.config import Parallelism5DConfig, get_5d_config


# =============================================================================
# GPU Configuration for Rust
# =============================================================================

# Use same A100-80GB for Rust
import modal

RUST_GPU_CONFIG = modal.gpu.A100(count=1, size="80GB")


# =============================================================================
# Rust CUDA Runner Class
# =============================================================================


class RustCUDARunner:
    """
    Rust CUDA runner with NCCL support.

    This runner:
    - Builds the Rust crate with CUDA features
    - Runs training via cargo
    - Syncs checkpoints bi-directionally
    """

    def __init__(
        self,
        crate_dir: str | Path,
        parallelism: Parallelism5DConfig,
        checkpoint_dir: str = "/checkpoints",
        data_dir: str = "/data",
    ):
        self.crate_dir = Path(crate_dir)
        self.parallelism = parallelism
        self.checkpoint_dir = checkpoint_dir
        self.data_dir = data_dir
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

    def build(self, release: bool = True) -> bool:
        """
        Build the Rust crate with CUDA features.

        Args:
            release: Build in release mode

        Returns:
            True if build succeeded
        """
        cmd = ["cargo", "build", "--features", "cuda"]
        if release:
            cmd.append("--release")

        print(f"[Rank {self.rank}] Building Rust crate: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            cwd=self.crate_dir,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"[Rank {self.rank}] Build failed:")
            print(result.stderr)
            return False

        print(f"[Rank {self.rank}] Build succeeded")
        return True

    def create_config_file(self, config_dict: dict[str, Any]) -> Path:
        """
        Create a JSON config file for the Rust binary.

        Args:
            config_dict: Training configuration

        Returns:
            Path to config file
        """
        config_path = self.crate_dir / "config.json"

        # Add distributed config
        config_dict["distributed"] = {
            "rank": self.rank,
            "world_size": self.world_size,
            "tensor_parallel_size": self.parallelism.tensor_parallel_size,
            "pipeline_parallel_size": self.parallelism.pipeline_parallel_size,
            "data_parallel_size": self.parallelism.data_parallel_size,
            "expert_parallel_size": self.parallelism.expert_parallel_size,
            "master_addr": os.environ.get("MASTER_ADDR", "localhost"),
            "master_port": int(os.environ.get("MASTER_PORT", 29500)),
        }

        # Add paths
        config_dict["checkpoint_dir"] = self.checkpoint_dir
        config_dict["data_dir"] = self.data_dir

        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        return config_path

    def run(
        self,
        stage: str,
        config_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Run Rust CUDA training.

        Args:
            stage: Training stage (pretrain, sft, grpo, distill)
            config_dict: Training configuration

        Returns:
            Training results
        """
        # Build first
        if not self.build(release=True):
            return {
                "rank": self.rank,
                "status": "build_failed",
                "metrics": {},
            }

        # Create config file
        config_path = self.create_config_file(config_dict)

        # Run the binary
        binary_path = self.crate_dir / "target" / "release" / "deepseek_from_scratch_in_rust"

        cmd = [
            str(binary_path),
            stage,
            "--config", str(config_path),
        ]

        print(f"[Rank {self.rank}] Running: {' '.join(cmd)}")

        # Set NCCL environment
        env = os.environ.copy()
        env.update({
            "RANK": str(self.rank),
            "WORLD_SIZE": str(self.world_size),
            "LOCAL_RANK": "0",
            "NCCL_DEBUG": os.environ.get("NCCL_DEBUG", "WARN"),
            "CUDA_VISIBLE_DEVICES": "0",
        })

        start_time = time.time()

        result = subprocess.run(
            cmd,
            cwd=self.crate_dir,
            capture_output=True,
            text=True,
            env=env,
        )

        elapsed = time.time() - start_time

        if result.returncode != 0:
            print(f"[Rank {self.rank}] Training failed:")
            print(result.stderr)
            return {
                "rank": self.rank,
                "status": "failed",
                "error": result.stderr,
                "elapsed_time": elapsed,
            }

        print(f"[Rank {self.rank}] Training completed in {elapsed:.1f}s")
        print(result.stdout)

        # Sync checkpoints (rank 0 only)
        if self.rank == 0:
            sync_checkpoints_to_volume()
            print(f"[Rank 0] Synced checkpoints to volume")

        # Parse metrics from stdout (Rust outputs JSON on last line)
        try:
            lines = result.stdout.strip().split("\n")
            metrics = json.loads(lines[-1]) if lines else {}
        except json.JSONDecodeError:
            metrics = {"raw_output": result.stdout}

        return {
            "rank": self.rank,
            "status": "success",
            "metrics": metrics,
            "elapsed_time": elapsed,
        }


# =============================================================================
# Modal Functions
# =============================================================================


@app.function(
    image=rust_cuda_image,
    gpu=RUST_GPU_CONFIG,
    secrets=[secrets],
    volumes={
        "/checkpoints": checkpoints_volume,
        "/data": data_volume,
    },
    timeout=86400,
    retries=RETRY_POLICY,
)
def rust_worker(
    rank: int,
    world_size: int,
    stage: str,
    config_dict: dict[str, Any],
    parallelism_dict: dict[str, Any],
    master_addr: str,
    master_port: int,
) -> dict[str, Any]:
    """
    Single Rust CUDA worker.

    Args:
        rank: Global rank
        world_size: Total workers
        stage: Training stage
        config_dict: Training config
        parallelism_dict: 5D parallelism config
        master_addr: NCCL master address
        master_port: NCCL master port

    Returns:
        Training results
    """
    # Set environment
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    # Reconstruct parallelism config
    parallelism = Parallelism5DConfig(
        tensor_parallel_size=parallelism_dict["tensor_parallel_size"],
        pipeline_parallel_size=parallelism_dict["pipeline_parallel_size"],
        data_parallel_size=parallelism_dict["data_parallel_size"],
        expert_parallel_size=parallelism_dict["expert_parallel_size"],
        sequence_parallel_size=parallelism_dict["sequence_parallel_size"],
    )

    # Clone and build the Rust crate
    # In Modal, we need to copy the crate from the mounted volume or include it in the image
    crate_dir = Path("/app/Deepseek-from-scratch-in-rust")

    # For now, assume the crate is copied to /app during image build or mounted
    # This could be done with modal.Mount or by including in the image

    runner = RustCUDARunner(
        crate_dir=crate_dir,
        parallelism=parallelism,
        checkpoint_dir="/checkpoints",
        data_dir="/data",
    )

    return runner.run(stage, config_dict)


@app.function(
    image=rust_cuda_image,
    gpu=RUST_GPU_CONFIG,
    secrets=[secrets],
    timeout=3600,
)
def build_rust() -> dict[str, Any]:
    """
    Build the Rust crate with CUDA features.

    Returns:
        Build result
    """
    crate_dir = Path("/app/Deepseek-from-scratch-in-rust")

    cmd = ["cargo", "build", "--release", "--features", "cuda"]

    print(f"Building: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        cwd=crate_dir,
        capture_output=True,
        text=True,
    )

    return {
        "success": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


@app.function(
    image=rust_cuda_image,
    secrets=[secrets],
    timeout=86400,
)
def train_rust(
    config_path: str | None = None,
    stage: str = "pretrain",
    model_size: str = "tiny",
    max_steps: int = 1000,
) -> dict[str, Any]:
    """
    Run Rust CUDA training on 10 GPUs.

    Args:
        config_path: Path to config JSON
        stage: Training stage (pretrain, sft, grpo, distill)
        model_size: Model size preset
        max_steps: Maximum training steps

    Returns:
        Training results
    """
    print("=" * 60)
    print("DeepSeek Rust CUDA Training")
    print("=" * 60)

    # Load or create config
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            config_dict = json.load(f)
    else:
        # Create config dict for Rust
        config_dict = {
            "model": {
                "size": model_size,
                "d_model": 256 if model_size == "tiny" else 512,
                "num_layers": 4 if model_size == "tiny" else 6,
                "num_heads": 4 if model_size == "tiny" else 8,
                "vocab_size": 8000 if model_size == "tiny" else 32000,
            },
            "training": {
                "max_steps": max_steps,
                "batch_size": 16,
                "learning_rate": 3e-4,
                "warmup_steps": 500,
            },
        }

    # 5D parallelism
    parallelism = get_5d_config(tp=2, pp=1, dp=5, ep=1, sp=1)
    parallelism_dict = parallelism.to_dict()

    print(f"Stage: {stage}")
    print(f"Configuration: {model_size}, {max_steps} steps")
    print(f"5D Parallelism: TP={parallelism.tensor_parallel_size}, "
          f"PP={parallelism.pipeline_parallel_size}, "
          f"DP={parallelism.data_parallel_size}")

    # Master address
    master_addr = "deepseek-from-scratch-rust-worker-0.modal.internal"
    master_port = 29500

    # Spawn workers
    start_time = time.time()

    worker_args = [
        (rank, parallelism.total_gpus, stage, config_dict, parallelism_dict, master_addr, master_port)
        for rank in range(parallelism.total_gpus)
    ]

    results = list(rust_worker.starmap(worker_args))

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s")

    # Aggregate
    successful = sum(1 for r in results if r.get("status") == "success")

    return {
        "elapsed_time": elapsed,
        "successful": successful,
        "total": len(results),
        "results": results,
    }


@app.local_entrypoint()
def main(
    config_path: str | None = None,
    stage: str = "pretrain",
    model_size: str = "tiny",
    max_steps: int = 1000,
):
    """Local entrypoint."""
    result = train_rust.remote(
        config_path=config_path,
        stage=stage,
        model_size=model_size,
        max_steps=max_steps,
    )
    print(f"\nResult: {json.dumps(result, indent=2)}")
