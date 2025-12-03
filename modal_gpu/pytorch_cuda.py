"""
PyTorch CUDA Test Runner for Modal
===================================

Simple wrapper script to test Modal's distributed_trainer directly.
For production use, run ray_pipeline locally with backend=MODAL_GPU.

Usage:
    # Test the distributed trainer directly
    uv run modal run modal_gpu/pytorch_cuda.py --max-steps 100
    
    # For production training, use ray_pipeline with MODAL_GPU backend:
    uv run python -m ray_pipeline.cli run --backend modal_gpu --max-steps 10000
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

# Import the distributed trainer app
from modal_gpu.distributed_trainer import (
    app,
    train_single_gpu,
    checkpoint_volume,
    training_volume,
)


@app.local_entrypoint()
def main(
    model_size: str = "tiny",
    max_steps: int = 100,
    data_path: str = "/data/stories/train",
):
    """
    Test the distributed trainer on Modal.
    
    Args:
        model_size: Model size preset (tiny, small, medium, large)
        max_steps: Maximum training steps
        data_path: Path to training data in Modal volume
    """
    print("=" * 60)
    print("DeepSeek Distributed Trainer - Test Run")
    print("=" * 60)
    
    print(f"\nModel size: {model_size}")
    print(f"Max steps: {max_steps}")
    print(f"Data path: {data_path}")
    
    # Model config based on size
    configs = {
        "tiny": {
            "hidden_size": 256,
            "num_layers": 4,
            "num_attention_heads": 4,
            "num_kv_heads": 2,
            "intermediate_size": 512,
            "vocab_size": 32000,
            "max_position_embeddings": 512,
        },
        "small": {
            "hidden_size": 512,
            "num_layers": 8,
            "num_attention_heads": 8,
            "num_kv_heads": 4,
            "intermediate_size": 1024,
            "vocab_size": 32000,
            "max_position_embeddings": 1024,
        },
    }
    
    model_config = configs.get(model_size, configs["tiny"])
    
    training_config = {
        "batch_size": 4,
        "learning_rate": 1e-4,
        "max_steps": max_steps,
        "warmup_steps": min(10, max_steps // 10),
        "gradient_accumulation_steps": 2,
        "use_amp": True,
        "save_steps": max(50, max_steps // 10),
        "log_steps": 10,
        "max_grad_norm": 1.0,
        "weight_decay": 0.01,
    }
    
    print("\nStarting training on Modal...")
    start_time = time.time()
    
    result = train_single_gpu.remote(
        model_config=model_config,
        training_config=training_config,
        data_path=data_path,
        checkpoint_dir="/checkpoints/test",
    )
    
    elapsed = time.time() - start_time
    
    print(f"\nTotal time (including Modal overhead): {elapsed:.1f}s")
    print(f"\nResult: {json.dumps(result, indent=2, default=str)}")
    
    return result
