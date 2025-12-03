# DeepSeek from Scratch

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/rust-stable-orange.svg)](https://www.rust-lang.org/)

Educational implementations of **DeepSeek-V3.2** and **DeepSeek-R1** architectures in **Rust** (using Candle) and **Python** (using PyTorch/MLX).

This repository provides from-scratch implementations of the key innovations that make DeepSeek models state-of-the-art:

### 🧠 Attention Mechanisms
- **Multi-Query Attention (MQA)** - Single KV head for memory-efficient inference
- **Grouped-Query Attention (GQA)** - Balanced KV sharing across head groups
- **Multi-Head Latent Attention (MLA)** - Compressed KV cache for efficient inference
- **DeepSeek Sparse Attention (DSA)** - Hybrid local + dilated global attention patterns

### 🔀 Mixture of Experts
- **Standard MoE** - Top-k expert routing with load balancing
- **DeepSeek MoE** - Fine-grained experts with shared expert isolation
- **256-Expert MoE** - Hierarchical routing for massive expert scaling

### 🎯 Prediction & Quantization
- **Multi-Token Prediction (MTP)** - Predict multiple future tokens simultaneously
- **FP8 Mixed-Precision** - Low-precision training with dynamic scaling
- **FP8 Quantization** - Simulated 8-bit inference for deployment

### 🏋️ Training & Alignment
- **GRPO Training** - Group Relative Policy Optimization for RL
- **DPO Training** - Direct Preference Optimization
- **SFT Pipeline** - Supervised Fine-Tuning infrastructure
- **Knowledge Distillation** - Teacher-student model compression
- **Agent & Tool-Use Training** - Function calling and tool integration

### 🚀 Infrastructure
- **5D Parallelism** - Tensor, Pipeline, Data, Expert, and Sequence parallelism
- **ZeRO Optimization** - Memory-efficient distributed training
- **DeepSeek-R1 Reasoning** - Chain-of-thought reasoning with `<think>` tags
- **Modal Cloud GPUs** - Distributed training on A100/H100 GPUs

---

## 📖 Table of Contents

- [Quick Start](#-quick-start)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Training Guide](#-training-guide)
- [Performance Benchmarks](#-performance-benchmarks)
- [Project Structure](#-project-structure)
- [Architecture Documentation](#-architecture-documentation)
- [Development](#-development)
- [Contributing](#-contributing)
- [License](#-license)
- [References](#-references)

---

## 🚀 Quick Start

### Train a Model in 5 Minutes

```bash
# 1. Clone and setup
git clone https://github.com/DevJadhav/deepseek-from-scratch.git
cd DeepSeek-From-Scratch

# 2. Install dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh  # Install UV if needed
uv sync

# 3. Download training data
uv run python scripts/download_tinystories.py

# 4. Train! (Choose one option)

# Option A: Local MLX (Apple Silicon - fastest for local dev)
uv run python -m ray_pipeline.cli run-mlx --max-steps 1000

# Option B: Modal Cloud GPU (Recommended for production)
pip install modal && modal setup
python -m ray_pipeline.cli run-rust --gpus 3 --pp-size 3 --max-steps 3000

# Option C: Local PyTorch (CPU/CUDA)
uv run python -m ray_pipeline.cli run --backend pytorch --model-size tiny --max-steps 1000
```

### Run Demos & Benchmarks

```bash
# PyTorch demos (CUDA/MPS/CPU)
cd deepseek-from-scratch-python
uv run python src/deepseek/main.py

# MLX demos (Apple Silicon native)
uv run python mlx_impl/main.py
uv run python mlx_impl/benchmark.py

# Rust demos (Metal)
cd Deepseek-from-scratch-in-rust
cargo run --release
```

---

## 🛠️ Prerequisites

### System Requirements

- **macOS 12.3+** (for Metal/MPS) or **Linux with CUDA**
- **Apple Silicon (M1/M2/M3/M4)** recommended for best local performance
- **8GB+ RAM** recommended (16GB+ for larger models)

### Required Tools

| Tool | Purpose | Installation |
|------|---------|--------------|
| **Python 3.10+** | Python implementation | [python.org](https://www.python.org/downloads/) |
| **UV** | Fast Python package manager | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **Rust** | Rust implementation | [rustup.rs](https://rustup.rs/) |
| **Modal** (optional) | Cloud GPU training | `pip install modal && modal setup` |

---

## 📦 Installation

### Python Setup (Recommended)

```bash
cd DeepSeek-From-Scratch

# Install with UV (fastest)
uv sync

# Or install with all optional extras
uv sync --all-extras  # Includes MLX, CoreML, dev tools
```

**Alternative (pip):**
```bash
pip install torch numpy einops transformers
pip install mlx  # Optional: Apple Silicon only
pip install coremltools  # Optional: CoreML export
```

### Rust Setup

```bash
cd Deepseek-from-scratch-in-rust

# Build in release mode (required for Metal acceleration)
cargo build --release
```

---

## 🎓 Training Guide

### Training Data Setup

```bash
# Download TinyStories dataset
uv run python scripts/download_tinystories.py
# Data saved to: data/stories/
```

### Training Options

#### Option 1: Modal Cloud GPUs (Production Recommended)

Best for: Production training, large-scale experiments

```bash
# Setup Modal (one-time)
pip install modal
modal setup

# Rust backend (fastest - 13.5 steps/sec)
python -m ray_pipeline.cli run-rust --gpus 3 --pp-size 3 --max-steps 3000

# Python backend (more flexible - 10.2 steps/sec)
python -m ray_pipeline.cli run-python --gpus 3 --pp-size 3 --max-steps 3000

# Full time-sliced execution (alternates Rust/Python)
python -m ray_pipeline.cli run --time-sliced --gpus 3 --pp-size 3 --max-steps 3000
```

#### Option 2: Local MLX (Apple Silicon)

Best for: Local development, quick iterations on Mac

```bash
# Memory-conscious config
uv run python -m ray_pipeline.cli run-mlx --max-steps 1500 --batch-size 2 --d-model 128

# Full config
uv run python -m ray_pipeline.cli run --backend mlx --model-size tiny --max-steps 5000
```

#### Option 3: Local PyTorch (CPU/CUDA)

Best for: Linux with CUDA, debugging

```bash
uv run python -m ray_pipeline.cli run --backend pytorch --model-size tiny --max-steps 1000
```

### Training Pipeline Stages

The ray_pipeline orchestrates a complete training workflow:

```
DATA_PREP → PRETRAIN → SFT → GRPO → DISTILLATION → EXPORT
```

| Stage | Description |
|-------|-------------|
| **DATA_PREP** | Tokenize and shard dataset |
| **PRETRAIN** | MTP + MoE pretraining |
| **SFT** | Supervised Fine-Tuning (instruction tuning) |
| **GRPO** | Group Relative Policy Optimization (alignment) |
| **DISTILLATION** | Knowledge distillation (optional) |
| **EXPORT** | Save final model + config |

### 5D Parallelism Configuration

The framework implements DeepSeek-style 5D parallelism:

| Dimension | Description | Default |
|-----------|-------------|---------|
| **PP** (Pipeline) | Splits model layers across GPUs | 3 |
| **DP** (Data) | Replicates model, splits data | 1 |
| **TP** (Tensor) | Splits layers horizontally | 1 |
| **EP** (Expert) | Distributes MoE experts | 1 |
| **SP** (Sequence) | Splits long sequences | 1 |

**Pipeline Parallelism Architecture (PP=3):**

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│    GPU 0     │──▶│    GPU 1     │──▶│    GPU 2     │
│ Embed+L1-4   │   │   L5-8       │   │ L9-12+Head   │
└──────────────┘   └──────────────┘   └──────────────┘
       ▲                                     │
       └────────── Gradient Flow ◀───────────┘
```

### Model Export

```bash
# Export to GGUF format
uv run python scripts/export_gguf.py --checkpoint checkpoints/final

# Export to CoreML (iOS/macOS)
uv run python deepseek-from-scratch-python/export_coreml.py
```

### Run Inference

```bash
uv run python scripts/inference.py --checkpoint checkpoints/final --prompt "Once upon a time"
```

---

## 📊 Performance Benchmarks

### Training Benchmarks (3000 steps)

| Backend | Hardware | Time | Steps/sec | Final Loss |
|---------|----------|------|-----------|------------|
| **Rust+GPU** | 3× H100 80GB | ~4 min | **13.5** | **1.18** |
| **Python+GPU** | 3× H100 80GB | ~5 min | 10.2 | 1.37 |
| **MLX** | Apple M1/M2/M3 | ~15 min | 3.3 | 1.85 |

### Component Benchmarks (Apple Silicon)

**Test Config:** batch_size=4, seq_len=64, d_model=512

#### Attention Mechanisms

| Component | Rust (Metal) | Python (MPS) | MLX |
|-----------|-------------|--------------|-----|
| **MQA** | 11.75ms | 0.95ms | 0.73ms |
| **GQA** | 11.00ms | 0.54ms | 0.82ms |
| **MLA** | 10.74ms | 0.96ms | 0.97ms |

#### Mixture of Experts

| Component | Rust (Metal) | Python (MPS) | MLX |
|-----------|-------------|--------------|-----|
| **Standard MoE** | 5.94ms | 134.87ms | - |
| **DeepSeek MoE** | 4.97ms | 49.85ms | 2.53ms |

#### Training Operations

| Component | Rust (Metal) | Python (MPS) | MLX |
|-----------|-------------|--------------|-----|
| **GRPO Loss** | 0.04ms | 0.73ms | 0.66ms |
| **DPO Loss** | 0.01ms | 0.28ms | 1.08ms |
| **KD Loss** | 0.05ms | 0.61ms | 0.32ms |

### Running Benchmarks

```bash
# PyTorch benchmarks (CUDA/MPS/CPU)
cd deepseek-from-scratch-python
uv run python -m pytest tests/ -v

# MLX benchmarks (Apple Silicon native)
uv run python mlx_impl/benchmark.py

# Rust benchmarks (Metal)
cd Deepseek-from-scratch-in-rust
cargo run --release
```

---

## 📁 Project Structure

```
DeepSeek-From-Scratch/
├── README.md                    # This file
├── LICENSE                      # Apache 2.0 License
├── pyproject.toml               # Python dependencies
├── uv.lock                      # Locked dependencies
│
├── deepseek-from-scratch-python/
│   ├── src/deepseek/            # PyTorch implementation (CUDA/MPS/CPU)
│   │   ├── main.py              # Entry point
│   │   ├── model/               # Model components
│   │   │   ├── attention.py     # MQA, GQA
│   │   │   ├── mla.py           # MLA, DeepSeek Attention
│   │   │   ├── moe.py           # MoE implementations
│   │   │   ├── mtp.py           # Multi-Token Prediction
│   │   │   ├── transformer.py   # Full transformer model
│   │   │   └── ...
│   │   └── training/            # Training infrastructure
│   │
│   ├── mlx_impl/                # MLX implementation (Apple Silicon native)
│   │   ├── main.py              # Entry point
│   │   ├── benchmark.py         # Benchmarks
│   │   ├── attention.py         # MQA, GQA, MLA
│   │   ├── moe.py               # MoE implementations
│   │   ├── mtp.py               # Multi-Token Prediction
│   │   ├── grpo.py              # GRPO training
│   │   ├── r1.py                # DeepSeek-R1 reasoning
│   │   └── ...
│   └── tests/                   # Test suite
│
├── Deepseek-from-scratch-in-rust/  # Rust/Candle implementation (Metal)
│   ├── Cargo.toml               # Rust dependencies
│   └── src/
│       ├── main.rs              # Entry point
│       ├── model/               # Model components
│       └── training/            # Training infrastructure
│
├── ray_pipeline/                # Training orchestration
│   ├── cli.py                   # Command-line interface
│   ├── config.py                # Configuration
│   ├── workflow.py              # Ray Workflow DAG
│   ├── stages/                  # Pipeline stages
│   └── runners/                 # Backend runners
│
├── modal_gpu/                   # Modal cloud GPU integration
│   ├── app.py                   # Modal app definition
│   ├── config.py                # 5D parallelism config
│   └── distributed_trainer.py   # GPU training runner
│
├── docs/                        # Architecture documentation (22 files)
│
├── scripts/                     # Utility scripts
│   ├── download_tinystories.py  # Download training data
│   ├── export_gguf.py           # GGUF export
│   ├── inference.py             # Run inference
│   └── train_tiny.py            # Quick training script
│
└── configs/                     # Configuration files
```

---

## 📚 Architecture Documentation

The `docs/` directory contains in-depth explanations of all architectural components:

### Attention Mechanisms
- [Multi-Query Attention (MQA)](docs/01-multi-query-attention.md)
- [Grouped-Query Attention (GQA)](docs/02-grouped-query-attention.md)
- [Multi-Head Latent Attention (MLA)](docs/03-multi-head-latent-attention.md)
- [DeepSeek Attention](docs/04-deepseek-attention.md)

### Mixture of Experts
- [Standard MoE](docs/05-standard-moe.md)
- [DeepSeek MoE](docs/06-deepseek-moe.md)

### Prediction & Quantization
- [Multi-Token Prediction (MTP)](docs/07-multi-token-prediction.md)
- [FP8 Quantization](docs/08-fp8-quantization.md)

### Training & Alignment
- [GRPO](docs/09-grpo.md)
- [Training Infrastructure](docs/10-training-infrastructure.md)
- [Training Pipeline](docs/11-training-pipeline.md)
- [Post-Training: SFT & RLHF](docs/12-post-training.md)
- [Knowledge Distillation](docs/13-knowledge-distillation.md)

### Advanced Topics
- [V3.2 Architecture Summary](docs/14-v32-architecture.md)
- [5D Parallelism](docs/15-5d-parallelism.md)
- [ZeRO Optimization](docs/16-zero-optimization.md)
- [Sparse Attention](docs/17-deepseek-sparse-attention.md)

---

## 🔧 Development

### Running Tests

```bash
# Python tests
cd deepseek-from-scratch-python
uv run pytest

# Rust tests
cd Deepseek-from-scratch-in-rust
cargo test
```

### Code Formatting

```bash
# Python
uv run black .
uv run ruff check .

# Rust
cargo fmt
cargo clippy
```

### Type Checking

```bash
# Python
uv run mypy ray_pipeline/
```

---

## 🤝 Contributing

Contributions are welcome! Here are some areas of interest:

- Flash Attention integration
- KV-Cache implementation
- Real FP8 hardware kernels
- Distributed training improvements
- Model weight loading from HuggingFace
- Additional cloud GPU providers (RunPod, Lambda Labs)
- Documentation improvements

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## 📚 References

- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [DeepSeek-R1 Technical Report](https://arxiv.org/abs/2501.12948)
- [Candle ML Framework](https://github.com/huggingface/candle)
- [MLX Framework](https://github.com/ml-explore/mlx)
- [Modal Cloud Platform](https://modal.com/)
- [Ray Framework](https://www.ray.io/)

---

## ⭐ Acknowledgments

This project is for educational purposes, demonstrating the key architectural innovations in DeepSeek models. Special thanks to:

- DeepSeek AI for their open research and technical reports
- Hugging Face for the Candle framework
- Apple for the MLX framework
- The open-source ML community
