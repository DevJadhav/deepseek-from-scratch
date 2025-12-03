"""
Modal Runner for Ray Pipeline
==============================

This runner is used by ray_pipeline to execute training and inference
on Modal's GPU clusters. It spawns Modal containers for GPU workloads
while the pipeline orchestration runs locally.

Architecture Flow:
    1. ray_pipeline runs locally, orchestrates stages
    2. When training is needed, ModalRunner.run() is called
    3. ModalRunner spawns Modal GPU containers
    4. Training runs on Modal with distributed parallelism
    5. Checkpoints sync back to local filesystem
    6. ray_pipeline continues with next stage

This enables:
    - Local development and debugging
    - Cloud GPU scaling on demand
    - 5D parallelism for large models
    - Cost-effective training (pay per use)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ray_pipeline.config import Backend, PipelineConfig
from ray_pipeline.runners.base import BaseRunner, RunnerResult


class ModalRunner(BaseRunner):
    """
    Runner that executes training on Modal's GPU cluster.
    
    This runner is called by ray_pipeline stages when GPU training is needed.
    It handles:
    - Uploading data to Modal volumes
    - Spawning GPU containers
    - Running distributed training
    - Downloading checkpoints back to local
    
    Usage:
        # In ray_pipeline stage
        runner = ModalRunner(config, stage="pretrain")
        result = runner.run(
            dataset_uri="./data/train",
            pad_token_id=0,
        )
    """
    
    name = "modal"
    
    def __init__(self, config: PipelineConfig, stage: str):
        super().__init__(config)
        self.stage = stage
        self.backend = config.detect_backend()
        
        # Modal function to call based on scale
        self._modal_function = self._select_modal_function()
    
    def _select_modal_function(self) -> str:
        """Select appropriate Modal function based on model size and parallelism."""
        if self.config.distributed.num_workers <= 1:
            return "train_single_gpu"
        else:
            return "train_distributed"
    
    def _sync_data_to_modal(self, local_path: str) -> str:
        """Upload training data to Modal volume."""
        print(f"[ModalRunner] Syncing data to Modal: {local_path}")
        
        # Resolve relative paths
        local_path_resolved = Path(local_path)
        if not local_path_resolved.is_absolute():
            local_path_resolved = self._get_project_root() / local_path
        
        # Check if this is a data directory with stories structure
        # We've already synced data to /train and /valid on the Modal volume
        if local_path_resolved.exists():
            # Check for stories subdirectory
            train_dir = local_path_resolved / "stories" / "train"
            if train_dir.exists():
                print(f"[ModalRunner] Found training data at: {train_dir}")
                # Sync train data
                train_files = list(train_dir.glob("*.jsonl")) + list(train_dir.glob("*.parquet"))
                for f in train_files:
                    result = subprocess.run(
                        ["uv", "run", "modal", "volume", "put", "deepseek-training-data",
                         str(f), f"/train/{f.name}"],
                        capture_output=True,
                        text=True,
                        cwd=self._get_project_root(),
                    )
                    if result.returncode == 0:
                        print(f"[ModalRunner] Uploaded: {f.name} -> /train/")
                
                # Return the path as it appears in the container
                # Volume is mounted at /data, so /train in volume = /data/train in container
                return "/data/train"
            else:
                # Try direct upload
                print(f"[ModalRunner] Uploading directory: {local_path_resolved}")
                result = subprocess.run(
                    ["uv", "run", "modal", "volume", "put", "deepseek-training-data", 
                     str(local_path_resolved), "/data/"],
                    capture_output=True,
                    text=True,
                    cwd=self._get_project_root(),
                )
                if result.returncode != 0:
                    print(f"[ModalRunner] Warning: Data sync returned {result.returncode}")
                    print(f"[ModalRunner] stderr: {result.stderr}")
                else:
                    print(f"[ModalRunner] Data synced successfully")
                return f"/data/{local_path_resolved.name}"
        else:
            print(f"[ModalRunner] Warning: Local path doesn't exist: {local_path_resolved}")
            print("[ModalRunner] Will try to use existing Modal volume data at /data/train")
            return "/data/train"
    
    def _sync_checkpoints_from_modal(self, remote_path: str, local_path: str):
        """Download checkpoints from Modal volume to local."""
        print(f"[ModalRunner] Syncing checkpoints from Modal: {remote_path} -> {local_path}")
        
        Path(local_path).mkdir(parents=True, exist_ok=True)
        
        result = subprocess.run(
            ["uv", "run", "modal", "volume", "get", "deepseek-checkpoints", remote_path, local_path],
            capture_output=True,
            text=True,
            cwd=self._get_project_root(),
        )
        
        if result.returncode != 0:
            print(f"[ModalRunner] Warning: Checkpoint sync returned {result.returncode}")
            print(f"[ModalRunner] stderr: {result.stderr}")
    
    def run(
        self,
        dataset_uri: str,
        pad_token_id: int,
        training_config: dict[str, Any] | None = None,
        extra_config: dict[str, Any] | None = None,
    ) -> RunnerResult:
        """
        Execute training on Modal GPU cluster.
        
        This method:
        1. Uploads training data to Modal volume
        2. Calls Modal function for GPU training
        3. Downloads checkpoints to local filesystem
        4. Returns results to ray_pipeline
        """
        print("=" * 60)
        print(f"[ModalRunner] Starting {self.stage} on Modal")
        print("=" * 60)
        
        # Prepare configs
        model_config = asdict(self.config.model)
        
        if training_config is None:
            training_dict = asdict(self.config.training)
        else:
            training_dict = training_config
        
        distributed_config = asdict(self.config.distributed)
        
        # Sync data to Modal
        modal_data_path = self._sync_data_to_modal(dataset_uri)
        
        # Modal checkpoint path
        modal_checkpoint_dir = f"/checkpoints/{self.stage}"
        
        # Determine implementation
        implementation = extra_config.get("implementation", "python") if extra_config else "python"

        # Build Modal command
        modal_cmd = self._build_modal_command(
            model_config=model_config,
            training_config=training_dict,
            distributed_config=distributed_config,
            data_path=modal_data_path,
            checkpoint_dir=modal_checkpoint_dir,
            implementation=implementation,
        )
        
        print(f"[ModalRunner] Running Modal command...")
        print(f"  Function: {self._modal_function}")
        print(f"  Data: {modal_data_path}")
        print(f"  Checkpoints: {modal_checkpoint_dir}")
        
        # Execute Modal function from project root
        project_root = self._get_project_root()
        
        result = subprocess.run(
            modal_cmd,
            capture_output=True,
            text=True,
            cwd=project_root,  # Run from project root where modal_gpu/ is located
        )
        
        print(f"\n[ModalRunner] Modal output:")
        print(result.stdout)
        
        if result.returncode != 0:
            print(f"[ModalRunner] Modal error:")
            print(result.stderr)
            return RunnerResult(
                metrics={"error": result.stderr},
                checkpoint_path=None,
                artifacts={},
                extra={"stage": self.stage, "status": "failed"},
            )
        
        # Parse result from Modal output
        try:
            # Look for JSON result in output
            for line in result.stdout.split("\n"):
                if line.strip().startswith("{") and "status" in line:
                    modal_result = json.loads(line)
                    break
            else:
                modal_result = {"status": "completed"}
        except json.JSONDecodeError:
            modal_result = {"status": "completed"}
        
        # Sync checkpoints back to local
        local_checkpoint_dir = self.config.training.checkpoint_dir
        self._sync_checkpoints_from_modal(
            modal_checkpoint_dir,
            str(Path(local_checkpoint_dir) / self.stage),
        )
        
        return RunnerResult(
            metrics=modal_result.get("metrics_history", [{}])[-1] if modal_result.get("metrics_history") else {},
            checkpoint_path=str(Path(local_checkpoint_dir) / self.stage / "final"),
            artifacts={},
            extra={
                "stage": self.stage,
                "modal_result": modal_result,
            },
        )
    
    def _build_modal_command(
        self,
        model_config: dict,
        training_config: dict,
        distributed_config: dict,
        data_path: str,
        checkpoint_dir: str,
        implementation: str = "python",
    ) -> list[str]:
        """Build the modal run command."""
        
        if implementation == "rust":
            # For Rust, we pass the full config as a JSON string
            # Reconstruct the full config dictionary
            full_config = {
                "model": model_config,
                "training": training_config,
                "distributed": distributed_config,
                "data": {"dataset_uri": data_path},
                "stage": self.stage,
            }
            config_json = json.dumps(full_config)
            
            cmd = [
                "uv", "run", "modal", "run",
                "modal_gpu/distributed_trainer.py::train_rust",
                "--config-json", config_json,
                "--stage", self.stage,
            ]
        else:
            # Python implementation
            max_steps = training_config.get("max_steps", 1000)
            # Handle wave-specific training config
            if isinstance(max_steps, dict):
                max_steps = max_steps.get("max_steps", 1000)
            
            cmd = [
                "uv", "run", "modal", "run",
                "modal_gpu/distributed_trainer.py",
                "--mode", "train",
                "--max-steps", str(max_steps),
                "--data-path", data_path,
            ]
            
        return cmd
    
    def _get_project_root(self) -> Path:
        """Get the project root directory."""
        # Go up from ray_pipeline/runners/ to project root
        return Path(__file__).parent.parent.parent
