"""
PyTorch Runner for Ray Pipeline
================================

This runner integrates with the actual PyTorch implementation from:

- ``deepseek.model.transformer.DeepSeekModel`` - Main transformer model
- ``deepseek.model.mla.DeepSeekAttention`` - Multi-head Latent Attention
- ``deepseek.model.moe.DeepSeekMoE`` - Mixture of Experts layer
- ``deepseek.training.training.Trainer`` - Training loop utilities

It uses Ray Train's TorchTrainer for distributed training with:

- Automatic device selection (CUDA/MPS/CPU)
- Gradient accumulation
- Mixed precision training (CUDA only)
- Checkpointing and fault tolerance

Usage
-----
The runner is invoked by pipeline stages (PretrainStage, SFTStage, etc.)::

    from ray_pipeline.runners import PyTorchRunner
    from ray_pipeline.config import PipelineConfig, ModelSize

    config = PipelineConfig.from_size(ModelSize.SMALL)
    runner = PyTorchRunner(config, stage="pretrain")
    result = runner.run(dataset_uri="./data/train.parquet", pad_token_id=0)

Backend Selection
-----------------
The runner validates that the detected backend is PyTorch-compatible:

- ``Backend.PYTORCH_CUDA`` - NVIDIA GPU with CUDA
- ``Backend.PYTORCH_MPS`` - Apple Silicon with Metal Performance Shaders
- ``Backend.PYTORCH_CPU`` - CPU fallback

Imported Modules from Implementation
------------------------------------
The following modules are imported from ``deepseek-from-scratch-python/src/deepseek/``:

- ``deepseek.model.transformer.DeepSeekModel``
- ``deepseek.model.mla.DeepSeekAttention``
- ``deepseek.model.moe.DeepSeekMoE``
- ``deepseek.training.training.get_device``
"""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Dict, Optional

import ray
from ray.train import Checkpoint, CheckpointConfig, RunConfig
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

from ray_pipeline.config import Backend, PipelineConfig
from ray_pipeline.runners.base import BaseRunner, RunnerResult


class PyTorchRunner(BaseRunner):
    """
    Runs training using PyTorch + Ray Train.
    
    This runner wraps the actual DeepSeek PyTorch implementation and provides:
    
    - Distributed training via Ray Train
    - Automatic sharding of Ray Datasets
    - Mixed precision training on CUDA
    - Checkpoint management
    
    Attributes
    ----------
    name : str
        Runner identifier ("pytorch")
    stage : str
        Current pipeline stage (e.g., "pretrain", "sft")
    backend : Backend
        Detected PyTorch backend (CUDA/MPS/CPU)
    
    Examples
    --------
    >>> config = PipelineConfig.from_size(ModelSize.SMALL)
    >>> runner = PyTorchRunner(config, stage="pretrain")
    >>> result = runner.run(dataset_uri="./cache/train", pad_token_id=0)
    >>> print(result.metrics)
    """

    name = "pytorch"

    def __init__(self, config: PipelineConfig, stage: str):
        """
        Initialize the PyTorch runner.
        
        Parameters
        ----------
        config : PipelineConfig
            Pipeline configuration with model and training settings
        stage : str
            Pipeline stage name for logging and checkpointing
            
        Raises
        ------
        ValueError
            If the detected backend is not PyTorch-compatible
        """
        super().__init__(config)
        self.stage = stage
        self.backend = config.detect_backend()
        if self.backend not in {
            Backend.PYTORCH_CUDA,
            Backend.PYTORCH_MPS,
            Backend.PYTORCH_CPU,
        }:
            raise ValueError(f"PyTorchRunner cannot run on backend {self.backend}")

    # ------------------------------------------------------------------
    # Ray Train Loop - Imports actual DeepSeek implementation
    # ------------------------------------------------------------------
    @staticmethod
    def _train_loop_per_worker(train_config: Dict[str, Any]):
        """
        Training loop executed by each Ray Train worker.
        
        This function imports and uses the actual DeepSeek PyTorch implementation:
        
        - ``deepseek.model.transformer.DeepSeekModel`` - Main model class
        - ``deepseek.model.mla.DeepSeekAttention`` - Used inside DeepSeekLayer
        - ``deepseek.model.moe.DeepSeekMoE`` - Used inside DeepSeekLayer
        - ``deepseek.training.training.get_device`` - Device selection utility
        
        Parameters
        ----------
        train_config : Dict[str, Any]
            Configuration containing model/training params
        """
        import torch
        import torch.nn.functional as F
        from ray import train
        from ray.train import get_context
        
        # ============================================================
        # ACTUAL DEEPSEEK IMPLEMENTATION IMPORTS
        # These are the real modules from deepseek-from-scratch-python
        # ============================================================
        from deepseek.model.transformer import DeepSeekModel  # Main model
        from deepseek.model.mla import DeepSeekAttention      # MLA attention (used by DeepSeekLayer)
        from deepseek.model.moe import DeepSeekMoE            # MoE layer (used by DeepSeekLayer)
        from deepseek.training.training import get_device      # Device selection utility

        ctx = get_context()
        ds = train.get_dataset_shard("train")
        
        # Use the actual get_device() from deepseek.training.training
        device = get_device()

        model_cfg = train_config["model"]
        training_cfg = train_config["training"]
        pad_token_id = train_config.get("pad_token_id", 0)
        vocab_size = model_cfg["vocab_size"]

        # Build kwargs matching DeepSeekLayer signature from transformer.py
        model_kwargs = dict(
            d_model=model_cfg["d_model"],
            num_heads=model_cfg["num_heads"],
            d_latent=model_cfg["d_latent"],
            d_rope=model_cfg["d_rope"],
            d_hidden=int(model_cfg["d_model"] * model_cfg["moe_hidden_mult"]),
            num_experts=model_cfg["num_experts"],
            num_shared=model_cfg["num_shared_experts"],
            num_routed=model_cfg["num_experts"],
            top_k=model_cfg["top_k"],
            use_moe=True,
        )

        # ============================================================
        # INSTANTIATE ACTUAL DEEPSEEK MODEL
        # DeepSeekModel internally creates DeepSeekLayer which uses:
        # - DeepSeekAttention (MLA + RoPE)
        # - DeepSeekMoE (shared + routed experts)
        # ============================================================
        model = DeepSeekModel(
            vocab_size=vocab_size,
            num_layers=model_cfg["num_layers"],
            gradient_checkpointing=training_cfg["gradient_checkpointing"],
            **model_kwargs,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_cfg["learning_rate"],
            betas=(training_cfg["beta1"], training_cfg["beta2"]),
            weight_decay=training_cfg["weight_decay"],
        )

        scaler = torch.cuda.amp.GradScaler(enabled=training_cfg["use_amp"] and device.type == "cuda")
        global_step = 0
        max_steps = training_cfg["max_steps"]
        grad_accum = training_cfg["gradient_accumulation_steps"]

        dataset_iter = ds.iter_torch_batches(
            batch_size=training_cfg["batch_size"],
            dtypes={"input_ids": torch.long, "attention_mask": torch.long},
            local_shuffle_buffer_size=training_cfg["batch_size"] * 4,
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)

        for batch in dataset_iter:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.cuda.amp.autocast(enabled=training_cfg["use_amp"] and device.type == "cuda"):
                # DeepSeekModel.forward() returns logits
                logits = model(input_ids, mask=attention_mask)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, vocab_size),
                    shift_labels.view(-1),
                    ignore_index=pad_token_id,
                )
                loss = loss / grad_accum

            if training_cfg["use_amp"] and device.type == "cuda":
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (global_step + 1) % grad_accum == 0:
                if training_cfg["use_amp"] and device.type == "cuda":
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), training_cfg["max_grad_norm"]
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), training_cfg["max_grad_norm"]
                    )
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1

            train.report({"loss": loss.item() * grad_accum, "step": global_step})

            if global_step >= max_steps:
                break

        checkpoint = Checkpoint.from_dict({"model_state_dict": model.state_dict()})
        train.save_checkpoint(checkpoint=checkpoint)

    # ------------------------------------------------------------------
    # Runner interface
    # ------------------------------------------------------------------
    def run(
        self,
        dataset_uri: str,
        pad_token_id: int,
        training_config=None,
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> RunnerResult:
        use_gpu = self.backend == Backend.PYTORCH_CUDA and self.config.distributed.use_gpu
        
        # Handle PP=3 configuration from extra_config
        num_workers = self.config.distributed.num_workers
        pipeline_parallel_size = self.config.distributed.pipeline_parallel_size
        gpu_ids = None
        
        if extra_config:
            if "pipeline_parallel_size" in extra_config:
                pipeline_parallel_size = extra_config["pipeline_parallel_size"]
            if "gpu_ids" in extra_config:
                gpu_ids = extra_config["gpu_ids"]
                # Set CUDA_VISIBLE_DEVICES for device placement
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
        
        # For PP=3, we need 3 workers
        if pipeline_parallel_size > 1:
            num_workers = pipeline_parallel_size

        train_ds = ray.data.read_parquet(dataset_uri)

        if training_config is None:
            training_dict = asdict(self.config.training)
        else:
            training_dict = training_config
        
        # Add PP stage info for distributed training
        training_dict["pipeline_parallel_size"] = pipeline_parallel_size
        training_dict["pipeline_stage"] = extra_config.get("pipeline_stage", 0) if extra_config else 0

        trainer = TorchTrainer(
            self._train_loop_per_worker,
            train_loop_config={
                "model": asdict(self.config.model),
                "training": training_dict,
                "pad_token_id": pad_token_id,
                "stage": self.stage,
                "extra": extra_config or {},
            },
            scaling_config=ScalingConfig(
                num_workers=num_workers,
                use_gpu=use_gpu,
                resources_per_worker={
                    "CPU": self.config.distributed.cpus_per_worker,
                    "GPU": self.config.distributed.gpus_per_worker if use_gpu else 0,
                },
                # Placement strategy for PP stages across GPUs
                placement_strategy="SPREAD" if pipeline_parallel_size > 1 else "PACK",
            ),
            datasets={"train": train_ds},
        )

        result = trainer.fit()
        checkpoint_path = None
        if result.checkpoint:
            checkpoint_path = result.checkpoint.to_uri()

        return RunnerResult(
            metrics=result.metrics,
            checkpoint_path=checkpoint_path,
            artifacts={},
            extra={"stage": self.stage},
        )
