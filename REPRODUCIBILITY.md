# Reproducibility Guide

This document provides complete instructions for reproducing all results from the DeepSeek-V3 from-scratch implementation.

## ✅ Last Verification

**Date:** December 5, 2025  
**Platform:** macOS (Apple Silicon)  
**Package Manager:** uv v0.7.8  
**Python:** 3.12.10  

| Component | Status | Details |
|-----------|--------|---------|
| Python Tests | ✅ 1,221 passed | 50 skipped (CUDA-only) |
| Rust Tests | ✅ 302 passed | 17 ignored (benchmarks) |
| uv sync | ✅ Working | All extras install correctly |
| MLX Backend | ✅ Working | Apple Silicon native |
| PyTorch Backend | ✅ Working | MPS/CPU |
| Rust/Metal | ✅ Working | Release builds pass |

## Table of Contents

- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Environment Setup](#environment-setup)
- [Training Commands](#training-commands)
- [Random Seeds](#random-seeds)
- [Expected Results](#expected-results)
- [Checkpoint Verification](#checkpoint-verification)
- [Troubleshooting](#troubleshooting)

---

## Hardware Requirements

### Minimum Requirements

| Component | Specification |
|-----------|---------------|
| CPU | 8+ cores (Intel/AMD x86_64 or Apple Silicon) |
| RAM | 16 GB |
| GPU | NVIDIA GPU with 8GB+ VRAM (Ampere or newer recommended) |
| Storage | 50 GB SSD |
| OS | Ubuntu 22.04, macOS 13+, or Windows 11 with WSL2 |

### Recommended Requirements

| Component | Specification |
|-----------|---------------|
| CPU | 16+ cores |
| RAM | 64 GB |
| GPU | NVIDIA A100 40GB or H100 80GB |
| Storage | 500 GB NVMe SSD |
| Network | 100 Gbps (for distributed training) |

### Hardware-Specific Notes

#### Apple Silicon (M1/M2/M3/M4)
- Use MLX backend for optimal performance
- 16GB+ unified memory recommended
- 32GB+ for medium-scale experiments

#### NVIDIA GPUs
| GPU | Max Model Size | Recommended Precision |
|-----|---------------|----------------------|
| RTX 3090/4090 | 1B params | FP16 |
| A100 40GB | 7B params | BF16 |
| A100 80GB | 13B params | BF16 |
| H100 80GB | 13B+ params | FP8 |

#### Multi-GPU Training
| Configuration | Supported Parallelism |
|---------------|----------------------|
| 2x GPUs | Data Parallel, FSDP |
| 4x GPUs | Data Parallel, FSDP, Pipeline Parallel |
| 8x GPUs | Full 3D Parallelism (DP + PP + EP) |

---

## Software Requirements

### Exact Versions

These are the tested versions that guarantee reproducibility:

```
# Python environment
Python: 3.10+ (tested with 3.12.10)
uv: 0.7.8+

# Core dependencies (from uv.lock)
torch: 2.2.0+
transformers: 4.35.0+
hydra-core: 1.3.2
omegaconf: 2.3.0
ray: 2.9.0+

# MLX (Apple Silicon only)
mlx: 0.22.0+

# Development
pytest: 9.0.1
ruff: 0.4.0+

# Rust
Edition: 2021
candle: 0.8.2 (with metal/accelerate features)
```

### Verification Commands

```bash
# Verify Python version
python --version

# Verify uv
uv --version

# Verify installation works
uv sync --all-extras

# Run full test suite
uv run pytest tests/ -v

# Verify CUDA (if using GPU)
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Verify dependencies
uv pip list | grep -E "torch|transformers"

# Rust verification
cd rust-src && cargo build --release && cargo test
```

---

## Environment Setup

### Step 1: Clone Repository

```bash
git clone https://github.com/DevJadhav/deepseek-from-scratch.git
cd deepseek-from-scratch

# Verify commit hash for exact reproducibility
git rev-parse HEAD
# Expected: [specific commit hash for release]
```

### Step 2: Install UV Package Manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or ~/.zshrc on macOS
uv --version  # Should show 0.4.0+
```

### Step 3: Install Dependencies

```bash
# Standard installation
uv sync

# With CUDA support
uv sync --extra cuda

# With MLX support (Apple Silicon)
uv sync --extra mlx

# Full development environment
uv sync --all-extras
```

### Step 4: Download Training Data

```bash
# TinyStories dataset
uv run python scripts/download_tinystories.py

# Verify data
ls -la data/stories/
# Expected: ~500MB of .txt files
```

### Step 5: Verify Installation

```bash
# Run full test suite
uv run pytest tests/ -v

# Quick backend verification
uv run python -c "from src.deepseek.torch.model import attention; print('PyTorch OK')"
uv run python -c "from src.deepseek.mlx import attention; print('MLX OK')"

# Rust verification
cd rust-src && cargo test
```

---

## Training Commands

### Model Size Variants

#### Tiny Model (109M parameters)
Best for: Ablation studies, quick experiments

```bash
# Single GPU
uv run python -m deepseek.pipeline.cli run \
    --backend pytorch \
    --model-size tiny \
    --max-steps 5000 \
    --batch-size 8 \
    --seed 42

# MLX (Apple Silicon)
uv run python -m deepseek.pipeline.cli run \
    --backend mlx \
    --max-steps 5000 \
    --batch-size 4 \
    --seed 42
```

#### Small Model (350M parameters)
Best for: Serious experimentation

```bash
# Single A100
uv run python -m deepseek.pipeline.cli run \
    --backend pytorch \
    --model-size small \
    --max-steps 20000 \
    --batch-size 16 \
    --gradient-accumulation 4 \
    --seed 42

# Multi-GPU with FSDP
torchrun --nproc_per_node=4 \
    -m deepseek.pipeline.cli run \
    --backend pytorch \
    --model-size small \
    --max-steps 20000 \
    --use-fsdp \
    --seed 42
```

#### Medium Model (1B parameters)
Best for: Production validation

```bash
# 8x A100 with full parallelism
torchrun --nproc_per_node=8 \
    -m deepseek.pipeline.cli run \
    --backend pytorch \
    --model-size medium \
    --max-steps 50000 \
    --use-fsdp \
    --pipeline-parallel 2 \
    --expert-parallel 4 \
    --seed 42
```

### Ablation Studies

```bash
# Run all ablations with 3 seeds
uv run python scripts/ablation/run_all_ablations.py \
    --seeds 42,123,456 \
    --max-steps 2000 \
    --output-dir ./ablation_results

# Run specific ablation
uv run python scripts/ablation/run_attention_ablation.py \
    --seeds 42,123,456 \
    --max-steps 2000
```

### Evaluation

```bash
# Perplexity evaluation
uv run python scripts/evaluate.py \
    --checkpoint ./checkpoints/final \
    --eval-perplexity

# Full benchmark suite
uv run python scripts/evaluate.py \
    --checkpoint ./checkpoints/final \
    --full-benchmark
```

---

## Random Seeds

For reproducibility, we use the following default seeds:

| Purpose | Seeds |
|---------|-------|
| Primary experiments | 42 |
| Ablation studies | 42, 123, 456 |
| Extended validation | 42, 123, 456, 789, 1024 |

### Setting Seeds Programmatically

```python
import torch
import numpy as np
import random

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # For deterministic cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

---

## Expected Results

### Tiny Model (5000 steps)

| Metric | Expected | Acceptable Range |
|--------|----------|------------------|
| Final Loss | 2.85 | 2.75 - 2.95 |
| Perplexity | 17.3 | 15.5 - 19.0 |
| Throughput | 8000 tok/s | 6000 - 12000 |
| Peak Memory | 4.2 GB | 3.5 - 5.0 GB |

### Small Model (20000 steps)

| Metric | Expected | Acceptable Range |
|--------|----------|------------------|
| Final Loss | 2.45 | 2.35 - 2.55 |
| Perplexity | 11.6 | 10.5 - 12.8 |
| Throughput | 5000 tok/s | 4000 - 7000 |
| Peak Memory | 18 GB | 15 - 22 GB |

### Ablation Studies

See `ablation_results/ablation_report.md` for detailed expected results.

### Variance Reporting

All results should be reported as: **mean ± std (n=3)**

Example:
```
Perplexity: 17.3 ± 0.4 (n=3 seeds)
```

---

## Checkpoint Verification

### SHA256 Hashes

Verify checkpoint integrity with these hashes:

```bash
# Tiny model checkpoint
sha256sum checkpoints/tiny/final/model.safetensors
# Expected: [hash]

# Small model checkpoint
sha256sum checkpoints/small/final/model.safetensors
# Expected: [hash]
```

### Checkpoint Structure

```
checkpoints/
├── tiny/
│   └── final/
│       ├── model.safetensors      # Model weights
│       ├── optimizer.pt           # Optimizer state
│       ├── training_config.json   # Training configuration
│       └── training_state.json    # Step, LR, RNG state
├── small/
│   └── ...
└── medium/
    └── ...
```

### Loading Checkpoints

```python
from safetensors.torch import load_file
import json

# Load model
state_dict = load_file("checkpoints/tiny/final/model.safetensors")
model.load_state_dict(state_dict)

# Load training state
with open("checkpoints/tiny/final/training_state.json") as f:
    state = json.load(f)
    
print(f"Trained for {state['step']} steps")
print(f"Final LR: {state['learning_rate']}")
```

---

## Troubleshooting

### Common Issues

#### CUDA Out of Memory

```bash
# Reduce batch size
--batch-size 4

# Enable gradient checkpointing
--gradient-checkpointing

# Use gradient accumulation instead of larger batches
--gradient-accumulation 8 --batch-size 2
```

#### Flash Attention Not Available

```bash
# Check CUDA version (needs 11.6+)
nvcc --version

# Reinstall flash-attn
uv pip install flash-attn --no-build-isolation --force-reinstall
```

#### MLX Errors on Apple Silicon

```bash
# Update MLX to latest
uv pip install --upgrade mlx

# Check Metal availability
python -c "import mlx.core as mx; print(mx.metal.is_available())"
```

#### Training Loss NaN/Inf

```bash
# Enable loss scaling for FP16
--use-grad-scaler

# Reduce learning rate
--learning-rate 1e-5

# Use BF16 instead of FP16
--precision bf16
```

#### Reproducibility Differences

Ensure:
1. Same CUDA version and GPU architecture
2. Same PyTorch version
3. `torch.backends.cudnn.deterministic = True`
4. Same number of GPUs and batch size
5. Same random seed

### Getting Help

1. Check existing GitHub Issues
2. Open new issue with:
   - Full error traceback
   - Hardware specs (`nvidia-smi`, `uname -a`)
   - Software versions (`uv pip list`)
   - Exact command used
   - Expected vs actual behavior

---

## Compute Requirements

### Estimated GPU Hours

| Model Size | Single GPU | 4x GPUs | 8x GPUs |
|------------|------------|---------|---------|
| Tiny (5K steps) | 1 hr | 0.3 hr | 0.2 hr |
| Small (20K steps) | 8 hr | 2.5 hr | 1.5 hr |
| Medium (50K steps) | 40 hr | 12 hr | 7 hr |

### Estimated Cloud Costs

| Provider | GPU Type | Tiny | Small | Medium |
|----------|----------|------|-------|--------|
| AWS | A100 40GB | $4 | $32 | $160 |
| GCP | A100 80GB | $5 | $40 | $200 |
| Lambda | H100 80GB | $3 | $24 | $120 |

---

## Citation

If you use this implementation, please cite:

```bibtex
@software{deepseek_from_scratch,
  title = {DeepSeek-V3 From Scratch},
  author = {DevJadhav},
  year = {2024},
  url = {https://github.com/DevJadhav/deepseek-from-scratch},
  version = {0.2.0}
}
```
