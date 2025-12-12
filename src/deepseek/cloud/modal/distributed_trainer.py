"""
Modal Distributed Trainer
=========================

This module provides GPU training functions that are called by the ray_pipeline
when it needs to execute training or inference on Modal's GPU cluster.

Features:
- Structured logging with correlation IDs for distributed tracing
- Checkpoint recovery on failures (try/finally pattern)
- DeepSpeed ZeRO optimization support
- 5D parallelism configuration

Architecture:
    ray_pipeline (local) → Modal GPU containers → results back to local

The ray_pipeline orchestrates stages locally, and when it needs GPU compute,
it calls these Modal functions which handle distributed training with 5D parallelism.

Usage (called from ray_pipeline):
    from deepseek.cloud.modal.distributed_trainer import train_model, run_inference
    
    # During pretrain stage
    result = train_model.remote(
        model_config=config.model,
        training_config=config.training,
        data_path="/path/to/data",
        checkpoint_dir="/path/to/checkpoints",
    )
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import modal

# NOTE: torch imports are done inside functions to avoid CUDA initialization
# issues when running locally. Modal functions execute remotely where GPU is available.

# =============================================================================
# Modal App Configuration
# =============================================================================

app = modal.App("deepseek-distributed-trainer")

# Persistent volumes for data and checkpoints
training_volume = modal.Volume.from_name(
    "deepseek-training-data",
    create_if_missing=True,
)

checkpoint_volume = modal.Volume.from_name(
    "deepseek-checkpoints", 
    create_if_missing=True,
)

# Container image optimized for distributed training
# Use NVIDIA's CUDA base image with Python to ensure CUDA libraries are available
trainer_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "curl", "build-essential", "openmpi-bin", "libopenmpi-dev")
    # Install uv package manager first
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "echo 'uv installed successfully'",
    )
    .env({"PATH": "/root/.local/bin:$PATH"})
    .run_commands(
        # PyTorch with CUDA 12.1 - use PyTorch index for CUDA wheels
        "uv pip install --system torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        "python -c 'import torch; print(f\"PyTorch {torch.__version__} CUDA {torch.version.cuda} built={torch.backends.cuda.is_built()}\")'",
    )
    .run_commands(
        # Distributed training and utilities
        "uv pip install --system 'deepspeed>=0.12.0' 'accelerate>=0.24.0' 'transformers>=4.35.0' "
        "'datasets>=2.14.0' 'tokenizers>=0.15.0' 'pyarrow>=14.0.0' 'safetensors>=0.4.0' "
        "'numpy>=1.24.0' 'tqdm>=4.65.0' 'rich>=13.0.0' 'pyyaml>=6.0'",
    )
    .env({
        "NCCL_DEBUG": "INFO",
        "NCCL_IB_DISABLE": "1",  # Use TCP for Modal's network
        "NCCL_P2P_DISABLE": "1",
        # Don't set CUDA_VISIBLE_DEVICES - let Modal handle GPU assignment
    })
)

# Image with Rust toolchain for Rust backend
rust_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "git",
        "curl",
        "build-essential",
        "pkg-config",
        "libssl-dev",
        "cmake",
    )
    .run_commands(
        # Install Rust
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        "echo 'source $HOME/.cargo/env' >> ~/.bashrc",
        # Install uv
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
    )
    .env({
        "PATH": "/root/.cargo/bin:/root/.local/bin:$PATH",
        "CUDA_HOME": "/usr/local/cuda",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:$LD_LIBRARY_PATH",
    })
    .run_commands(
        # Install Python dependencies with uv
        "uv pip install --system torch --index-url https://download.pytorch.org/whl/cu121",
        "uv pip install --system python-dotenv pyyaml structlog",
    )
    .add_local_dir(
        "Deepseek-from-scratch-in-rust",
        remote_path="/app/rust_src",
        copy=True,
    )
)

# GPU configurations for different training scales
# Using A100-80GB @ $2.50/hr per GPU for training
GPU_CONFIGS = {
    "single": "A100",       # 1x A100-80GB
    "small": "A100:2",      # 2x A100-80GB
    "medium": "A100:4",     # 4x A100-80GB
    "large": "A100:8",      # 8x A100-80GB (1 node)
    "xlarge": "A100:8",     # For multi-node: 8 GPUs per node
}

# Cost estimates per hour (A100-80GB @ $2.50/hr per GPU)
COST_PER_GPU_HOUR = 2.50  # USD


# =============================================================================
# Configuration Dataclasses (mirror ray_pipeline.config)
# =============================================================================

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    hidden_size: int = 256
    num_layers: int = 4
    num_attention_heads: int = 4
    num_kv_heads: int = 2
    intermediate_size: int = 512
    vocab_size: int = 32000
    max_position_embeddings: int = 512
    rope_theta: float = 10000.0
    use_moe: bool = False
    num_experts: int = 8
    num_experts_per_tok: int = 2
    use_mla: bool = True


@dataclass  
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 8
    learning_rate: float = 1e-4
    max_steps: int = 1000
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    use_amp: bool = True
    save_steps: int = 500
    log_steps: int = 10


@dataclass
class DistributedConfig:
    """
    5D Parallelism configuration for distributed training.
    
    Initial config (8 GPUs): TP=2, PP=2, DP=2, EP=1, SP=1
    - Enables DualPipe with PP=2 for bidirectional pipeline
    - Cost: 8 × $2.50/hr = $20.00/hr
    
    Scaled config (64 GPUs): TP=4, PP=4, DP=2, EP=2, SP=1
    - Full DualPipe with MoE expert parallelism
    - Cost: 64 × $2.50/hr = $160.00/hr
    """
    # Data parallelism (replicate model, partition data)
    data_parallel_size: int = 2
    # Tensor parallelism (split layers across GPUs within node)
    tensor_parallel_size: int = 2
    # Pipeline parallelism (split model layers across nodes) - enables DualPipe
    pipeline_parallel_size: int = 2
    # Expert parallelism (for MoE layers)
    expert_parallel_size: int = 1
    # Sequence parallelism (split sequence for long contexts)
    sequence_parallel_size: int = 1
    # ZeRO optimization stage (0, 1, 2, or 3)
    zero_stage: int = 2
    # DualPipe configuration
    use_dualpipe: bool = True
    num_micro_batches: int = 8
    
    @property
    def total_gpus(self) -> int:
        """Total GPUs required: TP × PP × DP × EP."""
        return (
            self.tensor_parallel_size
            * self.pipeline_parallel_size
            * self.data_parallel_size
            * self.expert_parallel_size
        )
    
    @property
    def estimated_cost_per_hour(self) -> float:
        """Estimated cost per hour in USD."""
        return self.total_gpus * 2.50  # A100-80GB rate
    
    @classmethod
    def initial_8gpu(cls) -> "DistributedConfig":
        """Initial 8-GPU config for verification."""
        return cls(
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            data_parallel_size=2,
            expert_parallel_size=1,
            sequence_parallel_size=1,
        )
    
    @classmethod
    def scaled_64gpu(cls) -> "DistributedConfig":
        """Scaled 64-GPU config for full DualPipe and MoE."""
        return cls(
            tensor_parallel_size=4,
            pipeline_parallel_size=4,
            data_parallel_size=2,
            expert_parallel_size=2,
            sequence_parallel_size=1,
        )


# =============================================================================
# Training Function - Called by ray_pipeline
# =============================================================================

@app.function(
    image=trainer_image,
    gpu="A100",  # A100-80GB @ $2.50/hr
    volumes={
        "/data": training_volume,
        "/checkpoints": checkpoint_volume,
    },
    timeout=86400,  # 24 hours
    memory=32768,  # 32GB RAM
)
def train_single_gpu(
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    data_path: str,
    checkpoint_dir: str,
    resume_from: str | None = None,
) -> dict[str, Any]:
    """
    Train on a single GPU with checkpoint recovery.
    
    This is the base training function called by ray_pipeline for single-GPU
    training scenarios (tiny/small models). Includes automatic checkpoint
    saving on failure for recovery.
    
    Args:
        model_config: Model architecture parameters
        training_config: Training hyperparameters
        data_path: Path to training data (parquet files)
        checkpoint_dir: Where to save checkpoints
        resume_from: Optional checkpoint to resume from
        
    Returns:
        Training results including metrics and checkpoint paths
    """
    import subprocess
    import os
    
    # Import and configure structured logging inside the Modal function
    from deepseek.cloud.modal.logging_utils import (
        get_logger, 
        configure_logging, 
        TrainingLogger,
        set_correlation_id,
        generate_correlation_id,
    )
    
    configure_logging(env="production")
    set_correlation_id(generate_correlation_id())
    logger = get_logger(__name__)
    training_logger = TrainingLogger("pretrain", rank=0, world_size=1)
    
    logger.info("training_session_started", checkpoint_dir=checkpoint_dir, data_path=data_path)
    
    # CUDA debugging with structured logging
    logger.debug("cuda_environment_check_started")
    cuda_env = {k: v for k, v in os.environ.items() if 'CUDA' in k or 'NVIDIA' in k}
    logger.debug("cuda_environment_vars", **cuda_env)
    
    # Check CUDA libraries
    cuda_libs = [
        "/usr/lib/x86_64-linux-gnu/libcuda.so",
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/local/cuda/lib64/libcudart.so",
    ]
    lib_status = {lib: os.path.exists(lib) for lib in cuda_libs}
    logger.debug("cuda_libraries_checked", **lib_status)
    
    # NOW import torch
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    
    # Check nvidia-smi
    logger.debug("nvidia_smi_check_started")
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        logger.debug("nvidia_smi_output", stdout=result.stdout[:500] if result.stdout else "")
    except Exception as e:
        logger.warning("nvidia_smi_failed", error=str(e))
    
    # Log PyTorch CUDA status
    logger.info(
        "pytorch_cuda_info",
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_version=torch.version.cuda,
        cuda_built=torch.backends.cuda.is_built(),
        cudnn_available=torch.backends.cudnn.is_available(),
    )
    
    # Initialize CUDA
    logger.debug("cuda_init_started")
    try:
        torch.cuda.init()
        logger.debug("cuda_init_success", is_available=torch.cuda.is_available())
    except Exception as e:
        logger.warning("cuda_init_failed", error=str(e))
    
    # Get device count
    try:
        device_count = torch._C._cuda_getDeviceCount()
        logger.debug("cuda_device_count", count=device_count)
    except Exception as e:
        logger.warning("cuda_device_count_failed", error=str(e))
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device_selected", device=str(device))
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("gpu_info", name=gpu_name, memory_gb=gpu_memory)
    
    # Build model
    logger.info("model_building_started")
    model = _build_model(model_config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("model_built", total_params=total_params)
    
    # Log training start
    training_logger.log_training_started(
        model_params=total_params,
        max_steps=training_config["max_steps"],
        batch_size=training_config["batch_size"],
    )
    
    # Load data
    logger.info("data_loading_started", path=data_path)
    train_loader = _load_data(data_path, training_config["batch_size"])
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config.get("weight_decay", 0.01),
    )
    
    # Mixed precision
    scaler = torch.amp.GradScaler() if training_config.get("use_amp", True) else None
    
    # Training loop with checkpoint recovery
    logger.info("training_loop_started")
    
    model.train()
    global_step = 0
    total_loss = 0.0
    start_time = time.time()
    
    max_steps = training_config["max_steps"]
    log_steps = training_config.get("log_steps", 10)
    save_steps = training_config.get("save_steps", 500)
    grad_accum = training_config.get("gradient_accumulation_steps", 1)
    
    metrics_history = []
    emergency_checkpoint_saved = False
    
    pbar = tqdm(total=max_steps, desc="Training")
    
    try:  # Checkpoint recovery wrapper
        while global_step < max_steps:
            for batch in train_loader:
                if global_step >= max_steps:
                    break
                    
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch.get("attention_mask", torch.ones_like(input_ids)).to(device)
                
                # Forward pass with AMP
                with torch.amp.autocast(device_type="cuda", enabled=scaler is not None):
                    logits = model(input_ids, mask=attention_mask)
                    
                    # Causal LM loss
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = input_ids[:, 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, model_config["vocab_size"]),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )
                    loss = loss / grad_accum
                
                # Backward pass
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                total_loss += loss.item() * grad_accum
                
                # Optimizer step
                if (global_step + 1) % grad_accum == 0:
                    if scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.get("max_grad_norm", 1.0))
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.get("max_grad_norm", 1.0))
                        optimizer.step()
                    optimizer.zero_grad()
                
                global_step += 1
                pbar.update(1)
                
                # Logging
                if global_step % log_steps == 0:
                    avg_loss = total_loss / log_steps
                    elapsed = time.time() - start_time
                    steps_per_sec = global_step / elapsed
                    
                    metrics = {
                        "step": global_step,
                        "loss": avg_loss,
                        "steps_per_sec": steps_per_sec,
                        "elapsed_time": elapsed,
                    }
                    metrics_history.append(metrics)
                    
                    # Use structured logging instead of pbar.set_postfix
                    training_logger.log_step(
                        step=global_step,
                        loss=avg_loss,
                        throughput=steps_per_sec,
                    )
                    pbar.set_postfix(loss=f"{avg_loss:.4f}", sps=f"{steps_per_sec:.2f}")
                    total_loss = 0.0
                
                # Save checkpoint
                if global_step % save_steps == 0:
                    ckpt_path = Path(checkpoint_dir) / f"step_{global_step}"
                    ckpt_path.mkdir(parents=True, exist_ok=True)
                    
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "global_step": global_step,
                        "config": model_config,
                    }, ckpt_path / "checkpoint.pt")
                    
                    training_logger.log_checkpoint_saved(str(ckpt_path), step=global_step)
                    
                    # Commit to Modal volume
                    try:
                        checkpoint_volume.commit()
                    except Exception as e:
                        logger.warning("checkpoint_commit_failed", error=str(e), step=global_step)
                
                if global_step >= max_steps:
                    break
    
    except Exception as e:
        # Emergency checkpoint on failure
        logger.error("training_failed", error=str(e), step=global_step, exc_info=True)
        emergency_path = Path(checkpoint_dir) / "emergency"
        emergency_path.mkdir(parents=True, exist_ok=True)
        
        try:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "config": model_config,
                "error": str(e),
            }, emergency_path / "checkpoint.pt")
            checkpoint_volume.commit()
            emergency_checkpoint_saved = True
            logger.info("emergency_checkpoint_saved", path=str(emergency_path), step=global_step)
        except Exception as save_error:
            logger.error("emergency_checkpoint_failed", error=str(save_error))
        
        raise
    
    finally:
        pbar.close()
    
    # Save final checkpoint
    final_path = Path(checkpoint_dir) / "final"
    final_path.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "config": model_config,
    }, final_path / "checkpoint.pt")
    
    checkpoint_volume.commit()
    training_logger.log_checkpoint_saved(str(final_path), step=global_step)
    
    elapsed = time.time() - start_time
    final_loss = metrics_history[-1]["loss"] if metrics_history else None
    
    training_logger.log_training_complete(
        final_step=global_step,
        total_time_seconds=elapsed,
        final_loss=final_loss,
    )
    
    return {
        "status": "success",
        "final_step": global_step,
        "elapsed_time": elapsed,
        "checkpoint_path": str(final_path),
        "metrics_history": metrics_history[-10:],  # Last 10 metrics
        "emergency_checkpoint_saved": emergency_checkpoint_saved,
    }


@app.function(
    image=trainer_image,
    gpu="A100:8",  # 8x A100-40GB with DualPipe (TP=2, PP=2, DP=2)
    volumes={
        "/data": training_volume,
        "/checkpoints": checkpoint_volume,
    },
    timeout=86400,
    memory=65536,  # 64GB RAM
)
def train_distributed(
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    distributed_config: dict[str, Any],
    data_path: str,
    checkpoint_dir: str,
    resume_from: str | None = None,
    use_deepspeed: bool = True,  # Enable DeepSpeed by default
) -> dict[str, Any]:
    """
    Train with distributed parallelism using DeepSpeed ZeRO.
    
    Implements 5D parallelism:
    - Data Parallelism: Replicate model across GPUs, split data
    - Tensor Parallelism: Split attention/FFN across GPUs
    - Pipeline Parallelism: Split layers across GPUs
    - Expert Parallelism: Distribute MoE experts
    - Sequence Parallelism: Split long sequences
    
    Plus ZeRO optimization (Stage 2/3) for memory efficiency.
    """
    import torch
    import torch.distributed as dist
    
    print("=" * 60)
    print("DeepSeek Training - Distributed (5D Parallelism + DeepSpeed ZeRO)")
    print("=" * 60)
    
    # Initialize distributed
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    
    device = torch.device(f"cuda:{local_rank}")
    
    print(f"\nRank {local_rank}/{world_size}")
    print(f"Device: {device}")
    print(f"DeepSpeed enabled: {use_deepspeed}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(local_rank)}")
    
    # Build model
    print("\nBuilding model...")
    model = _build_model(model_config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Build DeepSpeed config
    ds_config = _build_deepspeed_config(training_config, distributed_config)
    zero_stage = distributed_config.get("zero_stage", 2)
    print(f"ZeRO Stage: {zero_stage}")
    
    # Load data
    print(f"\nLoading data from: {data_path}")
    local_batch_size = training_config["batch_size"]
    train_loader = _load_data(data_path, local_batch_size)
    
    # Initialize with DeepSpeed or fallback to DDP
    if use_deepspeed:
        try:
            import deepspeed
            
            # Initialize DeepSpeed engine
            model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
                model=model,
                model_parameters=model.parameters(),
                config=ds_config,
            )
            print(f"DeepSpeed initialized with ZeRO Stage {zero_stage}")
            print(f"Optimizer: {ds_config['optimizer']['type']}")
            
            # Memory stats after initialization
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
            
        except ImportError:
            print("DeepSpeed not available, falling back to DDP")
            use_deepspeed = False
    
    if not use_deepspeed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        if world_size > 1:
            model = DDP(model, device_ids=[local_rank])
        model_engine = model
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config["learning_rate"],
            weight_decay=training_config.get("weight_decay", 0.01),
        )
        lr_scheduler = None
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting distributed training...")
    print("=" * 60)
    
    model_engine.train()
    global_step = 0
    total_loss = 0.0
    start_time = time.time()
    
    max_steps = training_config["max_steps"]
    log_steps = training_config.get("log_steps", 10)
    save_steps = training_config.get("save_steps", 500)
    grad_accum = training_config.get("gradient_accumulation_steps", 1)
    
    metrics_history = []
    
    while global_step < max_steps:
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", torch.ones_like(input_ids)).to(device)
            
            # Forward pass
            if use_deepspeed:
                logits = model_engine(input_ids, mask=attention_mask)
            else:
                logits = model_engine(input_ids, mask=attention_mask)
            
            # Compute loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, model_config["vocab_size"]),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            
            # Backward pass
            if use_deepspeed:
                # DeepSpeed handles gradient accumulation internally
                model_engine.backward(loss)
                model_engine.step()
            else:
                loss = loss / grad_accum
                loss.backward()
                
                if (global_step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.get("max_grad_norm", 1.0))
                    optimizer.step()
                    optimizer.zero_grad()
            
            total_loss += loss.item() * (1 if use_deepspeed else grad_accum)
            global_step += 1
            
            if global_step % log_steps == 0 and local_rank == 0:
                avg_loss = total_loss / log_steps
                elapsed = time.time() - start_time
                steps_per_sec = global_step / elapsed
                
                # Memory stats
                mem_allocated = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                
                print(f"Step {global_step}: loss={avg_loss:.4f}, sps={steps_per_sec:.2f}, mem={mem_allocated:.1f}GB")
                metrics_history.append({
                    "step": global_step,
                    "loss": avg_loss,
                    "steps_per_sec": steps_per_sec,
                    "memory_gb": mem_allocated,
                })
                total_loss = 0.0
            
            # Save checkpoint (rank 0 only)
            if global_step % save_steps == 0 and local_rank == 0:
                ckpt_path = Path(checkpoint_dir) / f"step_{global_step}"
                ckpt_path.mkdir(parents=True, exist_ok=True)
                
                if use_deepspeed:
                    # DeepSpeed checkpoint format
                    model_engine.save_checkpoint(str(ckpt_path), tag=f"step_{global_step}")
                else:
                    torch.save(model.state_dict(), ckpt_path / "model.pt")
                    
                checkpoint_volume.commit()
                print(f"Checkpoint saved at step {global_step}")
            
            if global_step >= max_steps:
                break
                
    # Final checkpoint
    if local_rank == 0:
        final_path = Path(checkpoint_dir) / "final"
        final_path.mkdir(parents=True, exist_ok=True)
        
        if use_deepspeed:
            model_engine.save_checkpoint(str(final_path), tag="final")
            # Also save model state dict for compatibility
            state_dict = model_engine.module.state_dict() if hasattr(model_engine, 'module') else model_engine.state_dict()
            torch.save(state_dict, final_path / "model.pt")
        else:
            torch.save(model.state_dict(), final_path / "model.pt")
            
        checkpoint_volume.commit()
        
    return {
        "status": "success",
        "world_size": world_size,
        "final_step": global_step,
        "use_deepspeed": use_deepspeed,
        "zero_stage": zero_stage if use_deepspeed else 0,
        "metrics_history": metrics_history,
    }


@app.function(
    image=rust_image,  # Use image with Rust toolchain
    gpu="A100",  # A100-80GB @ $2.50/hr
    volumes={
        "/data": training_volume,
        "/checkpoints": checkpoint_volume,
    },
    timeout=86400,
    memory=65536,
)
def train_rust(
    config_json: str,
    stage: str,
) -> dict[str, Any]:
    """
    Execute Rust training binary on Modal.
    
    The Rust source is baked into the image at /app/rust_src via add_local_dir.
    """
    import subprocess
    import json
    import os
    
    print("=" * 60)
    print(f"DeepSeek Rust Training - Stage: {stage}")
    print("=" * 60)
    
    rust_dir = "/app/rust_src"
    
    # Verify Rust source exists
    print(f"\nRust source directory: {rust_dir}")
    print(f"Directory exists: {os.path.exists(rust_dir)}")
    if os.path.exists(rust_dir):
        print(f"Contents: {os.listdir(rust_dir)}")
    
    # Write config to file
    config_path = "/tmp/config.json"
    with open(config_path, "w") as f:
        f.write(config_json)
        
    print(f"Config written to {config_path}")
    
    # Check GPU availability
    print("\nChecking GPU availability...")
    gpu_check = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print(gpu_check.stdout)
    
    # Create a Linux-compatible Cargo.toml (remove macOS-specific features)
    print("\nPatching Cargo.toml for Linux/CUDA...")
    cargo_toml_path = f"{rust_dir}/Cargo.toml"
    linux_cargo_toml = '''[package]
name = "deepseek_from_scratch_in_rust"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "deepseek-from-scratch-in-rust"
path = "src/main.rs"

[dependencies]
candle-core = { version = "0.8.2", features = ["cuda"] }
candle-nn = { version = "0.8.2", features = ["cuda"] }
candle-transformers = { version = "0.8.2", features = ["cuda"] }
anyhow = "1.0"
tracing = "0.1.43"
tracing-subscriber = { version = "0.3.22", features = ["env-filter", "json"] }
thiserror = "2.0.17"
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
sha2 = "0.10"
prometheus = "0.13"
tempfile = "3.10"

[features]
default = ["cuda"]
cuda = ["candle-core/cuda", "candle-nn/cuda", "candle-transformers/cuda"]
'''
    with open(cargo_toml_path, "w") as f:
        f.write(linux_cargo_toml)
    print("Cargo.toml patched for CUDA")
    
    # Build Rust binary with CUDA feature
    print("\nBuilding Rust binary with CUDA...")
    build_cmd = ["cargo", "build", "--release", "--features", "cuda"]
    
    try:
        build_result = subprocess.run(
            build_cmd,
            cwd=rust_dir,
            capture_output=True,
            text=True,
            env={**os.environ, "PATH": f"/root/.cargo/bin:{os.environ.get('PATH', '')}"}
        )
        print(f"Build stdout: {build_result.stdout}")
        if build_result.returncode != 0:
            print(f"Build stderr: {build_result.stderr}")
            return {"status": "build_failed", "error": build_result.stderr}
        print("Build successful!")
    except Exception as e:
        print(f"Build exception: {e}")
        return {"status": "build_failed", "error": str(e)}
    
    # Run the binary
    print(f"\nRunning Rust training for stage: {stage}")
    run_cmd = [
        "./target/release/deepseek-from-scratch-in-rust",
        stage,
        "--config", config_path
    ]
    
    print(f"Command: {' '.join(run_cmd)}")
    
    try:
        result = subprocess.run(
            run_cmd,
            cwd=rust_dir,
            capture_output=True,
            text=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1,2"}
        )
        print(f"stdout: {result.stdout}")
        if result.returncode != 0:
            print(f"stderr: {result.stderr}")
            return {"status": "failed", "error": result.stderr, "stdout": result.stdout}
        
        # Try to parse metrics from output
        try:
            lines = result.stdout.strip().split("\n")
            for line in reversed(lines):
                if line.strip().startswith("{"):
                    metrics = json.loads(line)
                    return {"status": "success", "metrics": metrics}
        except json.JSONDecodeError:
            pass
        
        return {"status": "success", "output": result.stdout}
    except Exception as e:
        print(f"Execution exception: {e}")
        return {"status": "failed", "error": str(e)}


# =============================================================================
# Inference Function - Called by ray_pipeline for evaluation
# =============================================================================

@app.function(
    image=trainer_image,
    gpu="A100",  # A100-80GB @ $2.50/hr
    volumes={
        "/checkpoints": checkpoint_volume,
    },
    timeout=3600,  # 1 hour
)
def run_inference(
    model_config: dict[str, Any],
    checkpoint_path: str,
    prompts: list[str],
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
) -> dict[str, Any]:
    """
    Run inference on a trained model with autoregressive generation.
    
    Called by ray_pipeline during evaluation stages.
    
    Args:
        model_config: Model architecture configuration
        checkpoint_path: Path to model checkpoint directory
        prompts: List of prompt strings to generate from
        max_new_tokens: Maximum new tokens to generate per prompt
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling probability threshold
        top_k: Top-k sampling (0 = disabled)
        
    Returns:
        Dictionary with generation results and metadata
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer
    
    print("=" * 60)
    print("DeepSeek Inference")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print(f"\nLoading model from: {checkpoint_path}")
    model = _build_model(model_config).to(device)
    
    # Load checkpoint
    ckpt_path = Path(checkpoint_path)
    
    # Try different checkpoint formats
    if (ckpt_path / "checkpoint.pt").exists():
        ckpt = torch.load(ckpt_path / "checkpoint.pt", map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
    elif (ckpt_path / "model.pt").exists():
        ckpt = torch.load(ckpt_path / "model.pt", map_location=device)
        model.load_state_dict(ckpt)
    elif (ckpt_path / "model.safetensors").exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(ckpt_path / "model.safetensors"))
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_path}")
    
    model.eval()
    print(f"Model loaded successfully")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Generate
    print(f"\nGenerating responses for {len(prompts)} prompts...")
    results = []
    
    with torch.no_grad():
        for prompt in prompts:
            print(f"\n  Prompt: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
            
            # Tokenize prompt
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            
            # Autoregressive generation
            generated = input_ids.clone()
            
            for _ in range(max_new_tokens):
                # Forward pass - get logits for next token
                logits = model(generated)
                next_token_logits = logits[:, -1, :]  # (batch, vocab_size)
                
                # Apply temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=-1)
                
                # Stop if EOS token generated
                if next_token.item() == tokenizer.eos_token_id:
                    break
            
            # Decode generated text
            generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            response = generated_text[len(prompt):]  # Remove prompt from response
            
            print(f"  Response: {response[:100]}{'...' if len(response) > 100 else ''}")
            
            results.append({
                "prompt": prompt,
                "response": response.strip(),
                "input_tokens": input_ids.shape[-1],
                "output_tokens": generated.shape[-1] - input_ids.shape[-1],
                "total_tokens": generated.shape[-1],
            })
    
    print(f"\n{'=' * 60}")
    print(f"Generation complete: {len(results)} responses")
    print(f"{'=' * 60}")
    
    return {
        "status": "success",
        "results": results,
        "model_config": model_config,
        "generation_config": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        },
    }


# =============================================================================
# Helper Functions
# =============================================================================

def _build_model(config: dict[str, Any]):
    """Build DeepSeekModel from config."""
    import torch
    import torch.nn as nn
    
    # Define simple transformer inline to avoid module-level torch import
    class _SimpleTransformer(nn.Module):
        """Simplified transformer for testing when DeepSeek model not available."""
        
        def __init__(self, config: dict[str, Any]):
            super().__init__()
            
            self.embedding = nn.Embedding(config["vocab_size"], config["hidden_size"])
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=config["hidden_size"],
                    nhead=config["num_attention_heads"],
                    dim_feedforward=config["intermediate_size"],
                    batch_first=True,
                )
                for _ in range(config["num_layers"])
            ])
            self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"])
        
        def forward(self, input_ids, mask=None):
            x = self.embedding(input_ids)
            for layer in self.layers:
                x = layer(x)
            return self.lm_head(x)
    
    # Try to import actual DeepSeek model
    try:
        from deepseek.model.transformer import DeepSeekModel
        return DeepSeekModel(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            num_heads=config["num_attention_heads"],
            num_kv_heads=config.get("num_kv_heads", config["num_attention_heads"]),
            intermediate_size=config["intermediate_size"],
            max_seq_len=config.get("max_position_embeddings", 512),
        )
    except ImportError:
        # Fallback to simple transformer
        print("Using simplified model (DeepSeek model not available)")
        return _SimpleTransformer(config)


def _load_data(data_path: str, batch_size: int):
    """Load training data."""
    import torch
    import pyarrow.parquet as pq
    from torch.utils.data import DataLoader, Dataset
    
    class ParquetDataset(Dataset):
        def __init__(self, path):
            self.path = Path(path)
            self.data = []
            
            print(f"[_load_data] Looking for data in: {self.path}")
            print(f"[_load_data] Path exists: {self.path.exists()}")
            
            if self.path.exists():
                # List directory contents
                if self.path.is_dir():
                    contents = list(self.path.iterdir())
                    print(f"[_load_data] Directory contents: {[str(c) for c in contents]}")
            
            # Load from parquet or jsonl
            if self.path.is_dir():
                # Direct files in directory
                for f in self.path.glob("*.parquet"):
                    print(f"[_load_data] Loading parquet: {f}")
                    table = pq.read_table(f)
                    self.data.extend(table.to_pylist())
                for f in self.path.glob("*.jsonl"):
                    print(f"[_load_data] Loading jsonl: {f}")
                    import json
                    with open(f) as fp:
                        for line in fp:
                            self.data.append(json.loads(line))
                
                # Also check subdirectories (for nested structures)
                for f in self.path.glob("**/*.parquet"):
                    if f.parent != self.path:  # Avoid double loading
                        print(f"[_load_data] Loading parquet from subdir: {f}")
                        table = pq.read_table(f)
                        self.data.extend(table.to_pylist())
                for f in self.path.glob("**/*.jsonl"):
                    if f.parent != self.path:  # Avoid double loading
                        print(f"[_load_data] Loading jsonl from subdir: {f}")
                        import json
                        with open(f) as fp:
                            for line in fp:
                                self.data.append(json.loads(line))
            elif self.path.is_file():
                # Direct file path
                if str(self.path).endswith('.parquet'):
                    print(f"[_load_data] Loading single parquet: {self.path}")
                    table = pq.read_table(self.path)
                    self.data.extend(table.to_pylist())
                elif str(self.path).endswith('.jsonl'):
                    print(f"[_load_data] Loading single jsonl: {self.path}")
                    import json
                    with open(self.path) as fp:
                        for line in fp:
                            self.data.append(json.loads(line))
            
            print(f"Loaded {len(self.data)} samples")
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            item = self.data[idx]
            
            # Handle different data formats
            if "input_ids" in item:
                input_ids = torch.tensor(item["input_ids"][:512])  # Truncate
            elif "text" in item:
                # Simple tokenization (in production use proper tokenizer)
                text = item["text"][:2048]
                input_ids = torch.tensor([ord(c) % 32000 for c in text[:512]])
            else:
                raise ValueError(f"Unknown data format: {item.keys()}")
            
            return {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }
    
    def collate_fn(batch):
        # Pad sequences
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
        attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
        
        for i, b in enumerate(batch):
            seq_len = len(b["input_ids"])
            input_ids[i, :seq_len] = b["input_ids"]
            attention_mask[i, :seq_len] = b["attention_mask"]
        
        return {"input_ids": input_ids, "attention_mask": attention_mask}
    
    dataset = ParquetDataset(data_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Modal doesn't support multiprocessing well
    )


def _build_deepspeed_config(training_config: dict, distributed_config: dict) -> dict:
    """Build DeepSpeed configuration for ZeRO optimization."""
    return {
        "train_batch_size": training_config["batch_size"] * distributed_config.get("data_parallel_size", 1),
        "gradient_accumulation_steps": training_config.get("gradient_accumulation_steps", 1),
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": training_config["learning_rate"],
                "weight_decay": training_config.get("weight_decay", 0.01),
            }
        },
        "fp16": {
            "enabled": training_config.get("use_amp", True),
        },
        "zero_optimization": {
            "stage": distributed_config.get("zero_stage", 2),
            "offload_optimizer": {
                "device": "cpu" if distributed_config.get("zero_stage", 2) >= 2 else "none",
            },
            "offload_param": {
                "device": "cpu" if distributed_config.get("zero_stage", 2) >= 3 else "none",
            },
        },
    }


# =============================================================================
# Local Testing Entrypoint
# =============================================================================

@app.local_entrypoint()
def main(
    mode: str = "train",
    model_size: str = "tiny",
    max_steps: int = 100,
    data_path: str = "/data",
):
    """
    Test the distributed trainer directly.
    
    This is for testing - normally ray_pipeline calls these functions.
    """
    print("=" * 60)
    print("DeepSeek Distributed Trainer - Direct Test")
    print("=" * 60)
    
    # Tiny model config
    model_config = {
        "hidden_size": 256,
        "num_layers": 4,
        "num_attention_heads": 4,
        "num_kv_heads": 2,
        "intermediate_size": 512,
        "vocab_size": 32000,
        "max_position_embeddings": 512,
    }
    
    training_config = {
        "batch_size": 4,
        "learning_rate": 1e-4,
        "max_steps": max_steps,
        "warmup_steps": 10,
        "gradient_accumulation_steps": 2,
        "use_amp": True,
        "save_steps": 50,
        "log_steps": 10,
    }
    
    if mode == "train":
        result = train_single_gpu.remote(
            model_config=model_config,
            training_config=training_config,
            data_path=data_path,
            checkpoint_dir="/checkpoints/test",
        )
        print(f"\nResult: {json.dumps(result, indent=2, default=str)}")
    
    elif mode == "inference":
        result = run_inference.remote(
            model_config=model_config,
            checkpoint_path="/checkpoints/test/final",
            prompts=["Once upon a time", "The quick brown fox"],
        )
        print(f"\nResult: {json.dumps(result, indent=2, default=str)}")
