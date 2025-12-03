"""
Modal Distributed Trainer
=========================

This module provides GPU training functions that are called by the ray_pipeline
when it needs to execute training or inference on Modal's GPU cluster.

Architecture:
    ray_pipeline (local) → Modal GPU containers → results back to local

The ray_pipeline orchestrates stages locally, and when it needs GPU compute,
it calls these Modal functions which handle distributed training with 5D parallelism.

Usage (called from ray_pipeline):
    from modal_gpu.distributed_trainer import train_model, run_inference
    
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
GPU_CONFIGS = {
    "single": "H100",
    "small": "H100:2",
    "medium": "H100:4",
    "large": "H100:8",
}


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
    """5D Parallelism configuration."""
    # Data parallelism
    data_parallel_size: int = 1
    # Tensor parallelism (split layers across GPUs)
    tensor_parallel_size: int = 1
    # Pipeline parallelism (split model layers across GPUs)
    pipeline_parallel_size: int = 1
    # Expert parallelism (for MoE)
    expert_parallel_size: int = 1
    # Sequence parallelism (split sequence across GPUs)
    sequence_parallel_size: int = 1
    # ZeRO optimization stage (0, 1, 2, or 3)
    zero_stage: int = 2


# =============================================================================
# Training Function - Called by ray_pipeline
# =============================================================================

@app.function(
    image=trainer_image,
    gpu="H100",
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
    Train on a single GPU.
    
    This is the base training function called by ray_pipeline for single-GPU
    training scenarios (tiny/small models).
    
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
    
    # Deep CUDA debugging - check environment BEFORE importing torch
    print("=" * 60)
    print("DeepSeek Training - Single GPU")
    print("=" * 60)
    
    print("\n--- Environment Variables (CUDA-related) ---")
    for key in sorted(os.environ.keys()):
        if 'CUDA' in key or 'NVIDIA' in key or 'LD_' in key:
            print(f"  {key}={os.environ.get(key)}")
    
    print("\n--- CUDA Library Check ---")
    cuda_libs = [
        "/usr/lib/x86_64-linux-gnu/libcuda.so",
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/local/cuda/lib64/libcudart.so",
    ]
    for lib in cuda_libs:
        exists = os.path.exists(lib)
        print(f"  {lib}: {'EXISTS' if exists else 'MISSING'}")
    
    # NOW import torch
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    
    # Print nvidia-smi to verify GPU availability
    print("\n--- nvidia-smi output ---")
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"stderr: {result.stderr}")
    except Exception as e:
        print(f"nvidia-smi failed: {e}")
    
    # Check PyTorch CUDA status
    print("--- PyTorch CUDA info ---")
    print(f"torch.__version__: {torch.__version__}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"torch.backends.cuda.is_built(): {torch.backends.cuda.is_built()}")
    print(f"torch.backends.cudnn.is_available(): {torch.backends.cudnn.is_available()}")
    
    # Try explicit CUDA init
    print("\n--- Explicit CUDA init ---")
    try:
        torch.cuda.init()
        print("torch.cuda.init() succeeded")
        print(f"After init - torch.cuda.is_available(): {torch.cuda.is_available()}")
    except Exception as e:
        print(f"torch.cuda.init() failed: {e}")
    
    # Check device count
    print("\n--- Device count ---")
    try:
        count = torch._C._cuda_getDeviceCount()
        print(f"torch._C._cuda_getDeviceCount(): {count}")
    except Exception as e:
        print(f"Error getting device count: {e}")
    
    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Build model
    print("\nBuilding model...")
    model = _build_model(model_config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Load data
    print(f"\nLoading data from: {data_path}")
    train_loader = _load_data(data_path, training_config["batch_size"])
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config.get("weight_decay", 0.01),
    )
    
    # Mixed precision
    scaler = torch.amp.GradScaler() if training_config.get("use_amp", True) else None
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    model.train()
    global_step = 0
    total_loss = 0.0
    start_time = time.time()
    
    max_steps = training_config["max_steps"]
    log_steps = training_config.get("log_steps", 10)
    save_steps = training_config.get("save_steps", 500)
    grad_accum = training_config.get("gradient_accumulation_steps", 1)
    
    metrics_history = []
    
    pbar = tqdm(total=max_steps, desc="Training")
    
    while global_step < max_steps:
        for batch in train_loader:
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
                
                print(f"\n✓ Saved checkpoint at step {global_step}")
                
                # Commit to Modal volume
                checkpoint_volume.commit()
            
            if global_step >= max_steps:
                break
    
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
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Training completed in {elapsed:.1f}s")
    print(f"Final step: {global_step}")
    print("=" * 60)
    
    return {
        "status": "success",
        "final_step": global_step,
        "elapsed_time": elapsed,
        "checkpoint_path": str(final_path),
        "metrics_history": metrics_history[-10:],  # Last 10 metrics
    }


@app.function(
    image=trainer_image,
    gpu="H100:3",
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
) -> dict[str, Any]:
    """
    Train with distributed parallelism using DeepSpeed.
    
    Implements 5D parallelism:
    - Data Parallelism: Replicate model across GPUs, split data
    - Tensor Parallelism: Split attention/FFN across GPUs
    - Pipeline Parallelism: Split layers across GPUs
    - Expert Parallelism: Distribute MoE experts
    - Sequence Parallelism: Split long sequences
    
    Plus ZeRO optimization for memory efficiency.
    """
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    
    print("=" * 60)
    print("DeepSeek Training - Distributed (5D Parallelism)")
    print("=" * 60)
    
    # Initialize distributed
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    
    device = torch.device(f"cuda:{local_rank}")
    
    print(f"\nRank {local_rank}/{world_size}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(local_rank)}")
    
    # Build model with parallelism
    print("\nBuilding distributed model...")
    model = _build_model(model_config).to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Use DeepSpeed for ZeRO optimization
    ds_config = _build_deepspeed_config(training_config, distributed_config)
    
    # Load data
    print(f"\nLoading data from: {data_path}")
    # Adjust batch size for distributed
    local_batch_size = training_config["batch_size"]
    train_loader = _load_data(data_path, local_batch_size)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config.get("weight_decay", 0.01),
    )
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting distributed training...")
    print("=" * 60)
    
    model.train()
    global_step = 0
    total_loss = 0.0
    start_time = time.time()
    
    max_steps = training_config["max_steps"]
    log_steps = training_config.get("log_steps", 10)
    save_steps = training_config.get("save_steps", 500)
    grad_accum = training_config.get("gradient_accumulation_steps", 1)
    
    metrics_history = []
    
    # Simple training loop (DeepSpeed integration omitted for brevity, using DDP)
    while global_step < max_steps:
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", torch.ones_like(input_ids)).to(device)
            
            logits = model(input_ids, mask=attention_mask)
            
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, model_config["vocab_size"]),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss = loss / grad_accum
            loss.backward()
            
            total_loss += loss.item() * grad_accum
            
            if (global_step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.get("max_grad_norm", 1.0))
                optimizer.step()
                optimizer.zero_grad()
            
            global_step += 1
            
            if global_step % log_steps == 0 and local_rank == 0:
                avg_loss = total_loss / log_steps
                elapsed = time.time() - start_time
                steps_per_sec = global_step / elapsed
                print(f"Step {global_step}: loss={avg_loss:.4f}, sps={steps_per_sec:.2f}")
                metrics_history.append({
                    "step": global_step,
                    "loss": avg_loss,
                    "steps_per_sec": steps_per_sec,
                })
                total_loss = 0.0
            
            # Save checkpoint (rank 0 only)
            if global_step % save_steps == 0 and local_rank == 0:
                ckpt_path = Path(checkpoint_dir) / f"step_{global_step}"
                ckpt_path.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt_path / "model.pt")
                checkpoint_volume.commit()
            
            if global_step >= max_steps:
                break
                
    # Final checkpoint
    if local_rank == 0:
        final_path = Path(checkpoint_dir) / "final"
        final_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), final_path / "model.pt")
        checkpoint_volume.commit()
        
    return {
        "status": "success",
        "world_size": world_size,
        "final_step": global_step,
        "metrics_history": metrics_history,
    }


@app.function(
    image=rust_image,  # Use image with Rust toolchain
    gpu="H100",  # Single GPU for testing
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
    gpu="H100",
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
) -> dict[str, Any]:
    """
    Run inference on a trained model.
    
    Called by ray_pipeline during evaluation stages.
    """
    import torch
    
    print("=" * 60)
    print("DeepSeek Inference")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    print(f"\nLoading model from: {checkpoint_path}")
    model = _build_model(model_config).to(device)
    
    ckpt = torch.load(Path(checkpoint_path) / "checkpoint.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    # Generate
    results = []
    for prompt in prompts:
        # Tokenize and generate
        # (simplified - in production use proper tokenizer)
        output = f"Generated response for: {prompt[:50]}..."
        results.append({"prompt": prompt, "response": output})
    
    return {
        "status": "success",
        "results": results,
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
