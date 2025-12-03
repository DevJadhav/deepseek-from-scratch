"""
Modal App Definition for DeepSeek Training
==========================================

Defines the Modal app with:
- A100-80GB preemptible GPUs
- Bi-directional checkpoint sync
- CUDA-enabled container image
- Secrets from environment

Usage
-----
Deploy the app::

    uv run modal deploy modal/app.py

Run training::

    uv run modal run modal/app.py::train
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

# =============================================================================
# Modal App Configuration
# =============================================================================

app = modal.App("deepseek-from-scratch")

# =============================================================================
# Secrets from Environment
# =============================================================================

# Load secrets from .env file
secrets = modal.Secret.from_dotenv(
    path=str(Path(__file__).resolve().parents[1] / ".env")
)

# =============================================================================
# Volumes for Checkpoint Persistence
# =============================================================================

# Create a persistent volume for checkpoints (bi-directional sync)
checkpoints_volume = modal.Volume.from_name(
    "deepseek-checkpoints",
    create_if_missing=True,
)

# Data volume for training data
data_volume = modal.Volume.from_name(
    "deepseek-data",
    create_if_missing=True,
)

# =============================================================================
# Container Image with CUDA + PyTorch + Rust
# =============================================================================

# Base image with CUDA 12.1
deepseek_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04",
        add_python="3.11",
    )
    # System dependencies
    .apt_install(
        "git",
        "curl",
        "build-essential",
        "pkg-config",
        "libssl-dev",
    )
    # Rust toolchain
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        "echo 'source $HOME/.cargo/env' >> ~/.bashrc",
    )
    .env({"PATH": "/root/.cargo/bin:$PATH"})
    # Python dependencies - install torch with CUDA 12.1 support
    .pip_install(
        # Core ML with CUDA support
        "torch>=2.1.0",
        "transformers>=4.35.0",
        "tokenizers>=0.15.0",
        "datasets>=2.14.0",
        "accelerate>=0.24.0",
        "safetensors>=0.4.0",
        # Ray for orchestration
        "ray[train,data]>=2.9.0",
        # Utilities
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "rich>=13.0.0",
        "typer>=0.9.0",
        "structlog>=25.5.0",
        "python-dotenv>=1.0.0",
        # NCCL for multi-GPU
        "nvidia-nccl-cu12>=2.18.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    # NCCL environment configuration
    .env({
        "NCCL_DEBUG": "WARN",
        "NCCL_IB_DISABLE": "0",
        "CUDA_VISIBLE_DEVICES": "all",
    })
)

# Rust-only image for Rust CUDA runner
rust_cuda_image = (
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
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    )
    .env({
        "PATH": "/root/.cargo/bin:$PATH",
        "CUDA_HOME": "/usr/local/cuda",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:$LD_LIBRARY_PATH",
    })
    .pip_install(
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0",
        "structlog>=25.5.0",
    )
)

# =============================================================================
# GPU Configuration
# =============================================================================

# H100 GPU specification (use string format as per Modal docs)
GPU_CONFIG = "H100"

# Preemptible/spot configuration with retries
RETRY_POLICY = modal.Retries(
    max_retries=3,
    initial_delay=60.0,
    backoff_coefficient=2.0,
)

# =============================================================================
# Helper Functions
# =============================================================================


def get_checkpoint_path() -> Path:
    """Get the checkpoint path inside the container."""
    return Path("/checkpoints")


def get_data_path() -> Path:
    """Get the data path inside the container."""
    return Path("/data")


def sync_checkpoints_to_volume():
    """
    Sync local checkpoints to Modal volume.
    
    Call this after each checkpoint save for bi-directional sync.
    """
    checkpoints_volume.commit()


def sync_checkpoints_from_volume():
    """
    Sync checkpoints from Modal volume to local.
    
    Call this at the start of training to resume from checkpoint.
    """
    checkpoints_volume.reload()


# =============================================================================
# Entry Point for Testing
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
def test_gpu_setup():
    """Test that GPU setup is working correctly."""
    import torch
    
    print("=" * 60)
    print("DeepSeek Modal GPU Setup Test")
    print("=" * 60)
    
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    
    # Check GPU info
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"  Compute capability: {props.major}.{props.minor}")
    
    # Check checkpoint volume
    checkpoint_path = get_checkpoint_path()
    print(f"\nCheckpoint path: {checkpoint_path}")
    print(f"Checkpoint path exists: {checkpoint_path.exists()}")
    
    # Check data volume
    data_path = get_data_path()
    print(f"Data path: {data_path}")
    print(f"Data path exists: {data_path.exists()}")
    
    # Check environment variables
    print("\nEnvironment:")
    print(f"  MODAL_TOKEN_ID set: {bool(os.getenv('MODAL_TOKEN_ID'))}")
    print(f"  NCCL_DEBUG: {os.getenv('NCCL_DEBUG', 'not set')}")
    
    print("=" * 60)
    print("Setup test complete!")
    print("=" * 60)
    
    return {
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "checkpoint_path": str(checkpoint_path),
        "data_path": str(data_path),
    }


@app.local_entrypoint()
def main():
    """Local entrypoint for testing."""
    result = test_gpu_setup.remote()
    print(f"Result: {result}")
