"""
GPU Orchestrator for Distributed Training
==========================================

Orchestrates 10x A100-80GB GPUs for distributed training with 5D parallelism.

5D Parallelism Configuration
----------------------------
- TP=2: Tensor parallel (split weights within GPU pairs)
- PP=1: Pipeline parallel (single stage for small models)
- DP=5: Data parallel (5 replicas processing different batches)
- EP=1: Expert parallel (all experts on same GPU)
- SP=1: Sequence parallel (no sequence splitting)

Usage
-----
Run distributed training::

    uv run modal run modal/gpu_runner.py::run_distributed_training

With custom config::

    uv run modal run modal/gpu_runner.py::run_distributed_training \\
        --config configs/tiny_mlx_full.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import modal

# Add project root to path for imports
_project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_project_root))

from modal_gpu.app import (
    GPU_CONFIG,
    RETRY_POLICY,
    app,
    checkpoints_volume,
    data_volume,
    deepseek_image,
    secrets,
    sync_checkpoints_to_volume,
)
from modal_gpu.config import ModalConfig, Parallelism5DConfig, get_5d_config

# =============================================================================
# GPU Worker Configuration
# =============================================================================


@dataclass
class WorkerConfig:
    """Configuration for a single GPU worker."""

    global_rank: int
    world_size: int
    tp_rank: int
    dp_rank: int
    pp_rank: int
    ep_rank: int
    sp_rank: int
    master_addr: str
    master_port: int
    checkpoint_dir: str
    data_dir: str


# =============================================================================
# GPU Orchestrator Class
# =============================================================================


class GPUOrchestrator:
    """
    Orchestrates distributed training across 10 GPUs with 5D parallelism.

    This class manages:
    - Spawning 10 GPU workers on Modal
    - Configuring NCCL for communication
    - Setting up 5D parallelism groups
    - Bi-directional checkpoint synchronization
    """

    def __init__(
        self,
        parallelism: Parallelism5DConfig | None = None,
        checkpoint_dir: str = "/checkpoints",
        data_dir: str = "/data",
    ):
        self.parallelism = parallelism or get_5d_config()
        self.checkpoint_dir = checkpoint_dir
        self.data_dir = data_dir
        self.num_gpus = self.parallelism.total_gpus

    def create_worker_configs(self, master_addr: str = "localhost") -> list[WorkerConfig]:
        """
        Create configuration for each GPU worker.

        Args:
            master_addr: Address of rank 0 for NCCL rendezvous

        Returns:
            List of WorkerConfig for each GPU
        """
        configs = []

        for rank in range(self.num_gpus):
            rank_mapping = self.parallelism.get_rank_mapping(rank)

            config = WorkerConfig(
                global_rank=rank,
                world_size=self.num_gpus,
                tp_rank=rank_mapping["tp_rank"],
                dp_rank=rank_mapping["dp_rank"],
                pp_rank=rank_mapping["pp_rank"],
                ep_rank=rank_mapping["ep_rank"],
                sp_rank=rank_mapping["sp_rank"],
                master_addr=master_addr,
                master_port=29500,
                checkpoint_dir=self.checkpoint_dir,
                data_dir=self.data_dir,
            )
            configs.append(config)

        return configs

    def get_nccl_env(self, worker_config: WorkerConfig) -> dict[str, str]:
        """
        Get NCCL environment variables for a worker.

        Args:
            worker_config: Configuration for the worker

        Returns:
            Environment variables dict
        """
        return {
            "RANK": str(worker_config.global_rank),
            "WORLD_SIZE": str(worker_config.world_size),
            "LOCAL_RANK": "0",  # One GPU per container
            "MASTER_ADDR": worker_config.master_addr,
            "MASTER_PORT": str(worker_config.master_port),
            "NCCL_DEBUG": os.getenv("NCCL_DEBUG", "WARN"),
            "NCCL_IB_DISABLE": os.getenv("NCCL_IB_DISABLE", "0"),
            # Tensor parallel group info
            "TP_RANK": str(worker_config.tp_rank),
            "TP_SIZE": str(self.parallelism.tensor_parallel_size),
            # Data parallel group info
            "DP_RANK": str(worker_config.dp_rank),
            "DP_SIZE": str(self.parallelism.data_parallel_size),
            # Pipeline parallel group info
            "PP_RANK": str(worker_config.pp_rank),
            "PP_SIZE": str(self.parallelism.pipeline_parallel_size),
        }


# =============================================================================
# Modal Function: GPU Worker
# =============================================================================


@app.function(
    image=deepseek_image,
    gpu=GPU_CONFIG,
    secrets=[secrets],
    volumes={
        "/checkpoints": checkpoints_volume,
        "/data": data_volume,
    },
    timeout=86400,  # 24 hours
    retries=RETRY_POLICY,
)
def gpu_worker(
    worker_config_dict: dict[str, Any],
    training_config_dict: dict[str, Any],
    nccl_env: dict[str, str],
) -> dict[str, Any]:
    """
    Single GPU worker function.

    This function runs on one A100-80GB GPU and participates in
    distributed training with other workers.

    Args:
        worker_config_dict: Worker configuration (serialized WorkerConfig)
        training_config_dict: Training configuration
        nccl_env: NCCL environment variables

    Returns:
        Training results and metrics
    """
    import torch
    import torch.distributed as dist

    # Set environment variables
    for key, value in nccl_env.items():
        os.environ[key] = value

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    print(f"[Rank {rank}/{world_size}] Starting GPU worker...")
    print(f"[Rank {rank}] TP_RANK={os.environ['TP_RANK']}, DP_RANK={os.environ['DP_RANK']}")

    # Initialize process group
    try:
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
            rank=rank,
            world_size=world_size,
        )
        print(f"[Rank {rank}] Process group initialized")
    except Exception as e:
        print(f"[Rank {rank}] Failed to initialize process group: {e}")
        # For testing, continue without distributed
        pass

    # Set device
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Get GPU info
    props = torch.cuda.get_device_properties(device)
    print(f"[Rank {rank}] GPU: {props.name}, Memory: {props.total_memory / 1024**3:.1f} GB")

    # Import training components
    try:
        from ray_pipeline.config import Backend, PipelineConfig
        from ray_pipeline.runners import PyTorchRunner

        # Create config from dict
        config = PipelineConfig.from_dict(training_config_dict)

        # Update distributed config
        config.distributed.num_workers = world_size
        config.distributed.tensor_parallel_size = int(os.environ["TP_SIZE"])
        config.distributed.data_parallel_size = int(os.environ["DP_SIZE"])
        config.distributed.pipeline_parallel_size = int(os.environ["PP_SIZE"])

        # Create runner
        runner = PyTorchRunner(config, stage="pretrain")

        # Run training
        result = runner.run(
            dataset_uri=worker_config_dict["data_dir"],
            pad_token_id=0,
        )

        metrics = result.metrics
        checkpoint_path = result.checkpoint_path

    except ImportError as e:
        print(f"[Rank {rank}] Import error: {e}")
        # Return dummy metrics for testing
        metrics = {
            "rank": rank,
            "status": "import_error",
            "error": str(e),
        }
        checkpoint_path = None
    except Exception as e:
        print(f"[Rank {rank}] Training error: {e}")
        metrics = {
            "rank": rank,
            "status": "error",
            "error": str(e),
        }
        checkpoint_path = None

    # Sync checkpoints to volume (bi-directional)
    if rank == 0 and checkpoint_path:
        print(f"[Rank {rank}] Syncing checkpoints to volume...")
        sync_checkpoints_to_volume()

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()

    return {
        "rank": rank,
        "metrics": metrics,
        "checkpoint_path": checkpoint_path,
    }


# =============================================================================
# Modal Function: Distributed Training Orchestration
# =============================================================================


@app.function(
    image=deepseek_image,
    secrets=[secrets],
    timeout=86400,
)
def run_distributed_training(
    config_path: str | None = None,
    model_size: str = "tiny",
    max_steps: int = 1000,
) -> dict[str, Any]:
    """
    Orchestrate distributed training across 10 GPUs.

    This function:
    1. Creates worker configurations with 5D parallelism
    2. Spawns 10 GPU workers in parallel
    3. Collects results and metrics
    4. Syncs checkpoints bi-directionally

    Args:
        config_path: Path to training config JSON (optional)
        model_size: Model size preset (tiny, small, medium, large)
        max_steps: Maximum training steps

    Returns:
        Aggregated training results
    """
    print("=" * 60)
    print("DeepSeek Distributed Training")
    print("=" * 60)

    # Load or create configuration
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            training_config = json.load(f)
        print(f"Loaded config from: {config_path}")
    else:
        # Create default config
        from ray_pipeline.config import Backend, ModelSize, PipelineConfig

        size_map = {
            "tiny": ModelSize.TINY,
            "small": ModelSize.SMALL,
            "medium": ModelSize.MEDIUM,
            "large": ModelSize.LARGE,
        }
        config = PipelineConfig.from_size(size_map.get(model_size, ModelSize.TINY))
        config.backend = Backend.PYTORCH_CUDA
        config.training.max_steps = max_steps
        training_config = config.to_dict()
        print(f"Created {model_size} config with {max_steps} steps")

    # Create orchestrator
    parallelism = get_5d_config(tp=2, pp=1, dp=5, ep=1, sp=1)
    orchestrator = GPUOrchestrator(parallelism=parallelism)

    print(f"\n5D Parallelism Configuration:")
    print(f"  Tensor Parallel (TP): {parallelism.tensor_parallel_size}")
    print(f"  Pipeline Parallel (PP): {parallelism.pipeline_parallel_size}")
    print(f"  Data Parallel (DP): {parallelism.data_parallel_size}")
    print(f"  Expert Parallel (EP): {parallelism.expert_parallel_size}")
    print(f"  Sequence Parallel (SP): {parallelism.sequence_parallel_size}")
    print(f"  Total GPUs: {parallelism.total_gpus}")

    # Create worker configs
    # Note: In Modal, we use the internal network for communication
    master_addr = "deepseek-from-scratch-gpu-worker-0.modal.internal"
    worker_configs = orchestrator.create_worker_configs(master_addr=master_addr)

    print(f"\nSpawning {len(worker_configs)} GPU workers...")

    # Spawn workers in parallel using Modal's starmap
    start_time = time.time()

    worker_args = []
    for wc in worker_configs:
        worker_config_dict = {
            "global_rank": wc.global_rank,
            "world_size": wc.world_size,
            "tp_rank": wc.tp_rank,
            "dp_rank": wc.dp_rank,
            "pp_rank": wc.pp_rank,
            "ep_rank": wc.ep_rank,
            "sp_rank": wc.sp_rank,
            "master_addr": wc.master_addr,
            "master_port": wc.master_port,
            "checkpoint_dir": wc.checkpoint_dir,
            "data_dir": wc.data_dir,
        }
        nccl_env = orchestrator.get_nccl_env(wc)
        worker_args.append((worker_config_dict, training_config, nccl_env))

    # Launch all workers
    results = list(gpu_worker.starmap(worker_args))

    elapsed_time = time.time() - start_time

    print(f"\nTraining completed in {elapsed_time:.1f}s")

    # Aggregate results
    successful = sum(1 for r in results if r.get("metrics", {}).get("status") != "error")
    failed = len(results) - successful

    print(f"\nResults: {successful} successful, {failed} failed")

    # Print per-rank metrics
    for result in results:
        rank = result.get("rank", "?")
        metrics = result.get("metrics", {})
        status = metrics.get("status", "unknown")
        if status == "error":
            print(f"  [Rank {rank}] ERROR: {metrics.get('error', 'unknown')}")
        else:
            loss = metrics.get("loss", "N/A")
            print(f"  [Rank {rank}] loss={loss}")

    return {
        "elapsed_time": elapsed_time,
        "successful_workers": successful,
        "failed_workers": failed,
        "results": results,
    }


# =============================================================================
# Local Entrypoint
# =============================================================================


@app.local_entrypoint()
def main(
    config_path: str | None = None,
    model_size: str = "tiny",
    max_steps: int = 1000,
):
    """
    Local entrypoint for running distributed training.

    Args:
        config_path: Path to training config JSON
        model_size: Model size preset
        max_steps: Maximum training steps
    """
    result = run_distributed_training.remote(
        config_path=config_path,
        model_size=model_size,
        max_steps=max_steps,
    )
    print(f"\nFinal result: {json.dumps(result, indent=2)}")
