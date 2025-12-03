# TinyStories Model Training Guide

This guide walks you through training a small language model on the TinyStories dataset using the DeepSeek-From-Scratch framework. The pipeline supports multiple backends including local MLX/PyTorch and **Modal cloud GPUs** for distributed training with **5D parallelism**.

## Performance Summary (3000 steps benchmark)

| Backend | Hardware | Parallelism | Time | Steps/sec | Final Loss |
|---------|----------|-------------|------|-----------|------------|
| **Rust+GPU** | 3× H100 80GB | PP=3 | ~4 min | **13.5** | **1.18** |
| **Python+GPU** | 3× H100 80GB | PP=3 | ~5 min | 10.2 | 1.37 |
| **MLX** | Apple M1/M2/M3 | - | ~15 min | 3.3 | 1.85 |

> **Recommendation:** Use **Rust+GPU** for production, **Python+GPU** for debugging, **MLX** for local development.

## Quick Start

### Prerequisites

```bash
# Install dependencies
uv sync

# For Modal GPU training (recommended for full training runs)
pip install modal
modal setup  # Authenticate with Modal
```

### Download Training Data

```bash
uv run python scripts/download_tinystories.py
```

This downloads the TinyStories dataset to `data/stories/`.

### Train the Model

#### Option 1: Modal Cloud GPUs (Recommended for Production)

Run training on Modal H100 GPUs with 5D parallelism (PP=3):

```bash
# Rust backend (fastest - 13.5 steps/sec, loss 1.18)
python -m ray_pipeline.cli run-rust --gpus 3 --pp-size 3 --max-steps 3000

# Python/PyTorch backend (more flexible - 10.2 steps/sec, loss 1.37)
python -m ray_pipeline.cli run-python --gpus 3 --pp-size 3 --max-steps 3000

# Full time-sliced execution (4 waves alternating Rust/Python)
python -m ray_pipeline.cli run --time-sliced --gpus 3 --pp-size 3 --max-steps 3000
```

**How it works:**
- Pipeline parallelism (PP=3) splits model across 3 H100 GPUs
- Each GPU holds ~1/3 of the model layers
- Forward pass flows GPU0 → GPU1 → GPU2
- Backward pass flows in reverse
- 80GB per GPU = 240GB total VRAM

**Legacy Commands:**
```bash
# Quick test (100 steps)
uv run python -m ray_pipeline.cli run --backend modal_gpu --model-size tiny --max-steps 100

# Full training (10,000 steps)
uv run python -m ray_pipeline.cli run --backend modal_gpu --model-size tiny --max-steps 10000
```

#### Option 2: Local MLX (Apple Silicon)

```bash
uv run python -m ray_pipeline.cli run --backend mlx --model-size tiny --max-steps 1000
```

#### Option 3: Local PyTorch (CPU/CUDA)

```bash
uv run python -m ray_pipeline.cli run --backend pytorch --model-size tiny --max-steps 1000
```

## Architecture Overview

### Pipeline Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    Local Machine                               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   ray_pipeline                           │  │
│  │  ┌────────────┬────────────┬────────────┬─────────────┐  │  │
│  │  │ DATA_PREP  │  PRETRAIN  │    SFT     │    GRPO     │  │  │
│  │  └────────────┴─────┬──────┴────────────┴─────────────┘  │  │
│  │                     │                                    │  │
│  │               ModalRunner                                │  │
│  └─────────────────────┼────────────────────────────────────┘  │
│                        │                                       │
└────────────────────────┼───────────────────────────────────────┘
                         │ Spawns GPU containers
                         ▼
┌────────────────────────────────────────────────────────────────┐
│                    Modal Cloud                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              distributed_trainer.py                      │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │  GPU Container (A100-80GB)                         │  │  │
│  │  │  - PyTorch + CUDA                                  │  │  │
│  │  │  - DeepSpeed (optional)                            │  │  │
│  │  │  - Training Loop                                   │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  │                                                          │  │
│  │  Volumes:                                                │  │
│  │  - /data (training data)                                 │  │
│  │  - /checkpoints (model checkpoints)                      │  │
│  │  - /outputs (training outputs)                           │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

### Components

1. **ray_pipeline** (Local Orchestrator)
   - Manages pipeline stages: DATA_PREP → PRETRAIN → SFT → GRPO → EXPORT
   - Selects appropriate runner based on `--backend` flag
   - Handles configuration and progress tracking

2. **ModalRunner** (`ray_pipeline/runners/modal_runner.py`)
   - Syncs training data to Modal volumes
   - Invokes Modal functions for GPU training
   - Downloads checkpoints and outputs after training

3. **distributed_trainer** (`modal_gpu/distributed_trainer.py`)
   - Modal app with GPU training functions
   - Supports single-GPU and multi-GPU distributed training
   - Uses DeepSpeed for efficient distributed training

### 5D Parallelism Support

The framework implements **DeepSeek-style 5D parallelism** for distributed training. Our production benchmarks ran with PP=3 on 3× H100 GPUs.

#### Parallelism Dimensions

| Dimension | Description | Production Value | Scaling Notes |
|-----------|-------------|------------------|---------------|
| **PP** (Pipeline) | Splits model layers across GPUs | **3** | Increases with model depth |
| **DP** (Data) | Replicates model, splits data | 1 | Increases with data throughput |
| **TP** (Tensor) | Splits layers horizontally | 1 | For very wide layers |
| **EP** (Expert) | Distributes MoE experts | 1 | For 256+ experts |
| **SP** (Sequence) | Splits long sequences | 1 | For 32k+ context |

#### Production Configuration (Tested)

```bash
# 3-GPU setup with PP=3 (our benchmark configuration)
python -m ray_pipeline.cli run-rust --gpus 3 --pp-size 3 --max-steps 3000
python -m ray_pipeline.cli run-python --gpus 3 --pp-size 3 --max-steps 3000
```

#### Pipeline Parallelism Architecture (PP=3)

```
┌────────────────────────────────────────────────────────────────┐
│                Modal Cloud (3× H100 80GB)                      │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   Forward Pass: ────────────────────────────────────────▶      │
│                                                                │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐       │
│   │    GPU 0     │   │    GPU 1     │   │    GPU 2     │       │
│   │  Stage 1/3   │──▶│  Stage 2/3   │──▶│  Stage 3/3   │       │
│   │ ┌──────────┐ │   │ ┌──────────┐ │   │ ┌──────────┐ │       │
│   │ │Embedding │ │   │ │Layers 5-8│ │   │ │Layers9-12│ │       │
│   │ │Layers 1-4│ │   │ │Attention │ │   │ │  LM Head │ │       │
│   │ │Attention │ │   │ │   MoE    │ │   │ │  Loss    │ │       │
│   │ │   MoE    │ │   │ └──────────┘ │   │ └──────────┘ │       │
│   │ └──────────┘ │   │              │   │              │       │
│   └──────────────┘   └──────────────┘   └──────────────┘       │
│                                                                │
│   ◀──────────────────────────────────────── Backward Pass      │
│                                                                │
│   Memory per GPU: ~20GB model + ~10GB activations              │
│   Communication: NVLink/PCIe for activation transfer           │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

#### Benchmark Results with 5D Parallelism

| Backend | Hardware | PP | Steps | Time | Steps/sec | Loss |
|---------|----------|------|-------|------|-----------|------|
| **Rust+GPU** | 3× H100 | 3 | 3000 | ~4 min | 13.5 | **1.18** |
| **Python+GPU** | 3× H100 | 3 | 3000 | ~5 min | 10.2 | 1.37 |
| **MLX** | M1/M2/M3 | 1 | 3000 | ~15 min | 3.3 | 1.85 |

#### Scaling Guidelines

| GPUs | Recommended Config | Total Memory | Use Case |
|------|-------------------|--------------|----------|
| 1 | PP=1, DP=1 | 80GB | Small models (<1B) |
| 3 | PP=3, DP=1 | 240GB | Medium models (1-7B) |
| 8 | PP=4, DP=2 | 640GB | Large models (7-13B) |
| 16 | PP=4, DP=2, TP=2 | 1.3TB | Very large models (13B+) |
| 64 | PP=4, DP=4, TP=2, EP=2 | 5.1TB | DeepSeek-scale (256 experts) |

#### Custom Parallelism Config

```json
{
  "distributed": {
    "num_workers": 3,
    "pipeline_parallel_size": 3,
    "data_parallel_size": 1,
    "tensor_parallel_size": 1,
    "expert_parallel_size": 1,
    "sequence_parallel_size": 1
  }
}
```

## Step-by-Step Guide

### Step 1: Verify Data

```bash
# Check dataset location
ls -la data/stories/train/
ls -la data/stories/valid/
```

Expected output:
```
data/stories/train/stories.jsonl
data/stories/valid/stories.jsonl
```

### Step 2: Configure Training

The default configuration is in `configs/tiny_mlx_quick.json`:

```json
{
  "model": {
    "vocab_size": 32000,
    "hidden_size": 256,
    "num_layers": 4,
    "num_heads": 4,
    "intermediate_size": 512
  },
  "training": {
    "batch_size": 4,
    "learning_rate": 3e-4,
    "max_steps": 1000,
    "gradient_accumulation_steps": 4
  }
}
```

### Step 3: Run Training

```bash
# Run with Modal GPUs
uv run python -m ray_pipeline.cli run --backend modal_gpu --model-size tiny --max-steps 1000

# Monitor progress - the pipeline will show:
# - Data sync progress
# - Training loss and metrics
# - Checkpoint saves
```

### Step 4: Monitor Training

Training metrics are logged to the console and saved to the outputs directory:

```bash
# Check outputs
ls -la outputs/

# Check checkpoints
ls -la checkpoints/
```

### Step 5: Run Inference

After training, test the model:

```bash
uv run python scripts/inference.py --checkpoint checkpoints/ray-tiny/final --prompt "Once upon a time"
```

## Backend Options

### Modal GPU (Recommended for Production)

```bash
uv run python -m ray_pipeline.cli run --backend modal_gpu --model-size tiny --max-steps 10000
```

**Features:**
- A100-80GB GPUs for fast training
- Automatic data sync to cloud
- Checkpoints saved locally after training
- Distributed training with DeepSpeed
- Pay-per-use pricing

### MLX (Apple Silicon)

```bash
uv run python -m ray_pipeline.cli run --backend mlx --model-size tiny --max-steps 1000
```

**Features:**
- Optimized for M1/M2/M3 Macs
- Uses unified memory
- Good for development and small experiments

### PyTorch (CPU/CUDA)

```bash
uv run python -m ray_pipeline.cli run --backend pytorch --model-size tiny --max-steps 1000
```

**Features:**
- Cross-platform compatibility
- Supports CUDA GPUs if available
- Standard PyTorch training loop

## Modal Cloud Training Details

### Authentication

```bash
# One-time setup
pip install modal
modal setup

# Verify authentication
modal token show
```

### Volume Management

Modal volumes persist data between runs:

| Volume | Purpose | Container Path |
|--------|---------|----------------|
| `deepseek-training-vol` | Training data | `/data` |
| `deepseek-checkpoint-vol` | Model checkpoints | `/checkpoints` |
| `deepseek-output-vol` | Training outputs | `/outputs` |

### GPU Selection

The default configuration uses H100-80GB GPUs with 5D parallelism (PP=3). You can modify `modal_gpu/distributed_trainer.py` or use CLI flags:

```bash
# Default: 3× H100 with PP=3
python -m ray_pipeline.cli run-rust --gpus 3 --pp-size 3 --max-steps 3000

# Single GPU (no parallelism)
python -m ray_pipeline.cli run-rust --gpus 1 --pp-size 1 --max-steps 3000

# 8 GPUs with PP=4, DP=2 (modify config.py)
python -m ray_pipeline.cli run-rust --gpus 8 --pp-size 4 --max-steps 3000
```

**Available GPU options in Modal:**
```python
# In modal_gpu/distributed_trainer.py:
# - modal.gpu.H100(count=3)      # H100-80GB (default, fastest)
# - modal.gpu.A100(count=3, size="80GB")  # A100-80GB
# - modal.gpu.A100(count=3)      # A100-40GB
# - modal.gpu.T4(count=1)        # T4-16GB (cheaper, no parallelism)
```

### Distributed Training

For multi-GPU distributed training:

```python
# In modal_gpu/distributed_trainer.py, the train_distributed function
# supports multiple GPUs with DeepSpeed ZeRO optimization

@app.function(
    gpu=modal.gpu.A100(count=4, size="80GB"),
    timeout=3600 * 4,
)
def train_distributed(config: dict) -> dict:
    # Uses DeepSpeed ZeRO-2 for memory-efficient training
    ...
```

## Troubleshooting

### Common Issues

#### 1. Modal Authentication Failed

```bash
# Re-authenticate
modal token new

# Verify
modal token show
```

#### 2. Data Not Found

```bash
# Ensure data is downloaded
uv run python scripts/download_tinystories.py

# Verify data location
ls -la data/stories/train/stories.jsonl
```

#### 3. GPU Not Detected

This shouldn't happen with Modal as GPUs are guaranteed. If you see issues:

```bash
# Test Modal GPU access
modal run modal_gpu/distributed_trainer.py::test_gpu
```

#### 4. Out of Memory

Reduce batch size or use gradient checkpointing:

```bash
# Use smaller batch size
uv run python -m ray_pipeline.cli run --backend modal_gpu --model-size tiny --max-steps 1000 --batch-size 2
```

#### 5. Training Interrupted

Checkpoints are saved periodically. Resume from the last checkpoint:

```bash
# Resume from checkpoint
uv run python -m ray_pipeline.cli run --backend modal_gpu --model-size tiny --resume --checkpoint-dir checkpoints/ray-tiny/
```

### Logging

Enable verbose logging:

```bash
# Set log level
export LOG_LEVEL=DEBUG

# Run with verbose output
uv run python -m ray_pipeline.cli run --backend modal_gpu --model-size tiny --max-steps 100 --verbose
```

## Advanced Pipeline Stages

### Full Pipeline

The complete pipeline includes multiple stages:

```bash
# Run full pipeline (all stages)
uv run python -m ray_pipeline.cli run --backend modal_gpu --stages all

# Or specify stages explicitly
uv run python -m ray_pipeline.cli run --backend modal_gpu --stages pretrain,sft,grpo
```

### Individual Stages

```bash
# Data preparation only
uv run python -m ray_pipeline.cli run --backend modal_gpu --stages data_prep

# Pretraining only
uv run python -m ray_pipeline.cli run --backend modal_gpu --stages pretrain

# Supervised fine-tuning
uv run python -m ray_pipeline.cli run --backend modal_gpu --stages sft

# GRPO reinforcement learning
uv run python -m ray_pipeline.cli run --backend modal_gpu --stages grpo
```

### Custom Configuration

Create a custom config file:

```json
{
  "model": {
    "vocab_size": 32000,
    "hidden_size": 512,
    "num_layers": 8,
    "num_heads": 8,
    "intermediate_size": 1024
  },
  "training": {
    "batch_size": 8,
    "learning_rate": 1e-4,
    "max_steps": 50000,
    "warmup_steps": 1000,
    "gradient_accumulation_steps": 8,
    "checkpoint_interval": 1000
  },
  "parallelism": {
    "tensor_parallel": 1,
    "pipeline_parallel": 1,
    "data_parallel": 4,
    "expert_parallel": 1,
    "sequence_parallel": 1
  }
}
```

Run with custom config:

```bash
uv run python -m ray_pipeline.cli run --backend modal_gpu --config my_config.json
```

## Output Structure

After training, your outputs will be organized as:

```
checkpoints/
└── ray-tiny/
    ├── step_100/
    │   ├── model.pt
    │   └── optimizer.pt
    ├── step_500/
    │   ├── model.pt
    │   └── optimizer.pt
    └── final/
        ├── model.pt
        ├── config.json
        └── training_state.json

outputs/
└── ray-tiny/
    ├── training_log.json
    ├── metrics.csv
    └── training_config.json
```

## Performance Tips

1. **Batch Size**: Larger batch sizes train faster but use more memory. A100-80GB can handle batch_size=16 or more.

2. **Gradient Accumulation**: Increase for effective larger batches without memory increase.

3. **Learning Rate**: Start with 3e-4 for small models, reduce for larger models.

4. **Checkpointing**: Save every 500-1000 steps to avoid losing progress.

5. **Mixed Precision**: Enabled by default on A100 for 2x speedup.

## Next Steps

After training your tiny model:

1. **Scale Up**: Increase model size with `--model-size small` or `--model-size medium`
2. **Fine-tune**: Use SFT stage with instruction-following data
3. **RLHF**: Use GRPO stage for preference optimization
4. **Export**: Export to GGUF format for llama.cpp inference
5. **Distillation**: Use knowledge distillation from larger models

## Resources

- [DeepSeek Paper](https://arxiv.org/abs/2401.02954)
- [TinyStories Paper](https://arxiv.org/abs/2305.07759)
- [Modal Documentation](https://modal.com/docs)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
