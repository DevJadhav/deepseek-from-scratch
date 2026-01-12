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
- [Docker Setup](#-docker-setup)
- [Training Guide](#-training-guide)
- [Ablation Studies](#-ablation-studies)
- [Performance Benchmarks](#-performance-benchmarks)
- [Project Structure](#-project-structure)
- [Architecture Documentation](#-architecture-documentation)
- [Reproducibility](#-reproducibility)
- [Development](#-development)
- [FAQ](#-frequently-asked-questions)
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
uv run python -m deepseek.pipeline.cli run --backend mlx --max-steps 1000

# Option B: Modal Cloud GPU (Recommended for production)
uv pip install modal && uv run modal setup
uv run modal run src/deepseek/cloud/modal/ray_cluster.py::run_pytorch --scale initial --max-steps 1000

# Option C: Local PyTorch (CPU/CUDA)
uv run python -m deepseek.pipeline.cli run --backend pytorch --model-size tiny --max-steps 1000
```

### Run Demos & Benchmarks

```bash
# PyTorch demos (CUDA/MPS/CPU)
uv run python -m deepseek.torch.main

# MLX demos (Apple Silicon native)
uv run python -m deepseek.mlx.main
uv run python -m deepseek.mlx.benchmark

# Rust demos (Metal)
cd rust-src
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
cd rust-src

# Build in release mode (Metal backend on macOS)
cargo build --release

# Build with CUDA support (Linux with NVIDIA GPU)
cargo build --release --features cuda

# Run tests
cargo test
```

---

## 🐳 Docker Setup

### Development Container (VS Code)

The easiest way to get started is with VS Code Dev Containers:

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop)
2. Install VS Code [Remote - Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
3. Open this folder in VS Code
4. Click "Reopen in Container" when prompted

### Docker Compose (Multi-Container)

```bash
# Development environment
docker compose up deepseek-dev

# Training with GPU
docker compose up deepseek-training

# Multi-GPU training (4 GPUs)
docker compose up deepseek-multi-gpu

# TensorBoard monitoring
docker compose up tensorboard
# Visit http://localhost:6006

# Jupyter notebooks
docker compose up jupyter
# Visit http://localhost:8888
```

### Manual Docker Build

```bash
# Build the image
docker build -t deepseek-from-scratch .

# Run with GPU support
docker run --gpus all -it deepseek-from-scratch

# Run with volume mounts
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/checkpoints:/app/checkpoints deepseek-from-scratch
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
uv pip install modal
uv run modal setup

# Run multi-GPU distributed training with PyTorch (8 A100 GPUs)
uv run modal run src/deepseek/cloud/modal/ray_cluster.py::run_pytorch --scale initial --max-steps 1000

# Run Rust backend verification (with CUDA)
uv run modal run src/deepseek/cloud/modal/ray_cluster.py::run_rust --scale initial --max-steps 100

# Pipeline CLI (alternative interface)
uv run python -m deepseek.pipeline.cli run --backend rust --gpus 3 --pp-size 3 --max-steps 3000
uv run python -m deepseek.pipeline.cli run --backend pytorch --gpus 3 --pp-size 3 --max-steps 3000
```

**GPU Configuration:** Standard config uses 8 A100-40GB GPUs with DualPipe (TP=2, PP=2, DP=2). For scaled runs (>8 GPUs), 
the framework automatically orchestrates sequential 8-GPU batches with checkpointing for fault tolerance.

#### Option 2: Local MLX (Apple Silicon)

Best for: Local development, quick iterations on Mac

```bash
# Memory-conscious config
uv run python -m deepseek.pipeline.cli run --backend mlx --max-steps 1500 --batch-size 2 --d-model 128

# Full config
uv run python -m deepseek.pipeline.cli run --backend mlx --model-size tiny --max-steps 5000
```

#### Option 3: Local PyTorch (CPU/CUDA)

Best for: Linux with CUDA, debugging

```bash
uv run python -m deepseek.pipeline.cli run --backend pytorch --model-size tiny --max-steps 1000
```

### Training Pipeline Stages

The pipeline orchestrates a complete training workflow:

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

## 🔬 Ablation Studies

We provide comprehensive ablation study infrastructure to analyze component contributions.

### Running Ablations

```bash
# Run all ablations
uv run python scripts/ablation/run_all_ablations.py --output-dir results/ablations

# Individual ablations
uv run python scripts/ablation/run_attention_ablation.py    # MLA vs GQA vs MHA
uv run python scripts/ablation/run_expert_ablation.py       # 8 vs 64 vs 256 experts
uv run python scripts/ablation/run_balancing_ablation.py    # Aux-loss-free vs aux loss
uv run python scripts/ablation/run_mtp_ablation.py          # MTP depth D=0,1,2,3
uv run python scripts/ablation/run_precision_ablation.py    # FP8 vs BF16 vs FP16
```

### Ablation Results Summary

| Study | Best Configuration | Key Finding |
|-------|-------------------|-------------|
| Attention | MLA | 14× KV cache compression, no quality loss |
| Experts | 256 with K=8 | Diminishing returns beyond 256 |
| Balancing | Aux-loss-free | Cleaner gradients, better convergence |
| MTP Depth | D=1 | 1.4× speculative decoding speedup |
| Precision | FP8 per-block | 2.4× throughput, minimal accuracy loss |

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
├── src/deepseek/                # Main Python package
│   ├── torch/                   # PyTorch implementation (CUDA/MPS/CPU)
│   │   ├── model/               # Model components (attention, moe, mla, transformer)
│   │   ├── training/            # Training infrastructure (grpo, sft, fsdp)
│   │   ├── kernels/             # Triton kernels
│   │   └── utils/               # Utilities
│   │
│   ├── mlx/                     # MLX implementation (Apple Silicon native)
│   │   ├── attention.py         # MQA, GQA, MLA
│   │   ├── moe.py               # MoE implementations
│   │   ├── grpo.py              # GRPO training
│   │   ├── r1.py                # DeepSeek-R1 reasoning
│   │   └── ane_impl/            # Apple Neural Engine optimizations
│   │
│   ├── pipeline/                # Ray training orchestration
│   │   ├── cli.py               # Command-line interface
│   │   ├── config.py            # Configuration
│   │   ├── workflow.py          # Ray Workflow DAG
│   │   ├── stages/              # Pipeline stages (pretrain, sft, grpo)
│   │   └── runners/             # Backend runners (mlx, pytorch, rust, modal)
│   │
│   ├── cloud/modal/             # Modal cloud GPU integration
│   │   ├── app.py               # Modal app definition
│   │   ├── config.py            # 5D parallelism config
│   │   └── distributed_trainer.py
│   │
│   └── common/                  # Shared utilities
│       └── tracking/            # Profiling and W&B integration
│
├── rust-src/                    # Rust/Candle implementation (Metal)
│   ├── Cargo.toml               # Rust dependencies
│   └── src/
│       ├── main.rs              # Entry point
│       ├── model/               # Model components
│       └── training/            # Training infrastructure
│
├── config/                      # Configuration files
│   ├── tiny_mlx_*.json          # MLX training configs
│   └── hydra/                   # Hydra configuration
│
├── tests/                       # Test suite
│   ├── torch/                   # PyTorch backend tests
│   ├── mlx/                     # MLX backend tests
│   ├── ane/                     # Apple Neural Engine tests
│   ├── pipeline/                # Pipeline tests
│   └── cloud/                   # Cloud integration tests
│
├── docs/                        # Architecture documentation (22+ files)
│
├── scripts/                     # Utility scripts
│   ├── download_tinystories.py  # Download training data
│   ├── export_gguf.py           # GGUF export
│   ├── inference.py             # Run inference
│   └── train_tiny.py            # Quick training script
│
├── monitoring/                  # Cost tracking and dashboards
│
└── checkpoints/                 # Model checkpoints (reproducibility examples)
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

### Blog Posts (Technical Deep-Dives)
- [Multi-Latent Attention Deep Dive](docs/blog/01_mla_deep_dive.md)
- [Auxiliary-Loss-Free Load Balancing](docs/blog/02_auxiliary_loss_free.md)
- [DualPipe Pipeline Parallelism](docs/blog/03_dualpipe_explained.md)
- [Expert Specialization Analysis](docs/blog/04_expert_specialization.md)
- [From Scratch to Production](docs/blog/05_production_lessons.md)

### Paper-Ready Materials
- [Architecture Diagrams](docs/paper/architecture.md)
- [Algorithm Pseudocode](docs/paper/pseudocode.md)
- [LaTeX Tables](docs/paper/tables.tex)

---

## 📋 Reproducibility

For complete reproduction instructions, see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

### Quick Reproduction

```bash
# 1. Setup environment
uv sync

# 2. Download data
uv run python scripts/download_tinystories.py

# 3. Train with specific seed for reproducibility
uv run python scripts/train_tiny.py --seed 42 --max-steps 1000

# 4. Run benchmarks
uv run python scripts/benchmark.py --config configs/tiny_test.json
```

### Expected Results (TinyStories, 1000 steps)

| Metric | Expected | Tolerance |
|--------|----------|-----------|
| Training Loss | ~2.5 | ±0.2 |
| Validation Loss | ~2.7 | ±0.2 |
| Throughput (M1) | ~3K tok/s | ±500 |
| Memory (tiny) | ~1GB | ±200MB |

---

## ✅ Verification Status

**Last Verified:** December 5, 2025

| Component | Status | Tests | Notes |
|-----------|--------|-------|-------|
| **Python (uv)** | ✅ Passing | 1,221 passed, 50 skipped | Full test suite |
| **Rust (Candle)** | ✅ Passing | 302 passed, 17 ignored | Metal backend |
| **PyTorch Backend** | ✅ Working | All tests pass | MPS/CPU |
| **MLX Backend** | ✅ Working | All tests pass | Apple Silicon |
| **ANE Backend** | ✅ Working | All tests pass | Neural Engine |
| **Triton Kernels** | ✅ Working | N/A | Requires CUDA |
| **CUDA Backend** | ✅ Working | N/A | Requires NVIDIA GPU |

**Package Manager:** `uv` v0.7.8  
**Python Version:** 3.12.10  
**Rust Edition:** 2021  

---

## 🔧 Development

### Running Tests

```bash
# Python tests (full suite)
uv run pytest tests/ -v

# Python tests by backend
uv run pytest tests/torch/ -v        # PyTorch backend
uv run pytest tests/mlx/ -v          # MLX backend
uv run pytest tests/ane/ -v          # Apple Neural Engine
uv run pytest tests/pipeline/ -v     # Pipeline orchestration

# Rust tests
cd rust-src
cargo test

# Rust tests with CUDA (on NVIDIA systems)
cd rust-src
cargo test --features cuda
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

## ❓ Frequently Asked Questions

### General Questions

**Q: What's the difference between PyTorch, MLX, and Rust implementations?**
A: 
- **PyTorch**: Most complete, supports CUDA/MPS/CPU, best for research
- **MLX**: Optimized for Apple Silicon, fastest on Mac
- **Rust**: Best performance on Metal, best for production deployment

**Q: Do I need a GPU to run this?**
A: No! All implementations support CPU. However, for training:
- Apple Silicon: MLX provides excellent performance
- NVIDIA GPU: PyTorch with CUDA is recommended
- Production: Rust with Metal or CUDA

**Q: How much memory do I need?**
A: For the tiny model (~10M params):
- Minimum: 4GB RAM
- Recommended: 8GB+ RAM
- Full training: 16GB+ RAM or GPU memory

### Training Questions

**Q: Why is my training loss not decreasing?**
A: Common causes:
1. Learning rate too high - try reducing by 10x
2. Data not properly tokenized - check data pipeline
3. Gradient explosion - enable gradient clipping

**Q: How do I resume training from a checkpoint?**
A:
```bash
uv run python scripts/train_tiny.py --resume checkpoints/step_500
```

**Q: How do I train on my own dataset?**
A: See the data preparation guide in `docs/11-training-pipeline.md`. Key steps:
1. Tokenize your data
2. Create training shards
3. Update config to point to your data

### Technical Questions

**Q: What is Multi-Latent Attention (MLA)?**
A: MLA compresses the KV cache by projecting keys and values to a lower-dimensional latent space before storage. This reduces memory by 14× compared to standard attention while maintaining quality. See `docs/03-multi-head-latent-attention.md`.

**Q: How does auxiliary-loss-free balancing work?**
A: Instead of adding a loss term that affects gradients, we use learnable biases that only affect routing decisions (not gating weights). After each step, biases are adjusted based on load. See `docs/blog/02_auxiliary_loss_free.md`.

**Q: Why use FP8 instead of FP16/BF16?**
A: FP8 provides:
- 2× memory reduction vs FP16
- 2-4× throughput improvement on modern hardware
- Minimal accuracy loss with per-block scaling

---

## 🤝 Contributing

Contributions are welcome! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

We also have a [Code of Conduct](CODE_OF_CONDUCT.md) that we expect all contributors to follow.

### Areas of Interest

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

## 📖 Citation

If you use this project in your research, please cite:

```bibtex
@software{deepseek_from_scratch,
  title={DeepSeek From Scratch: Educational Implementation of DeepSeek-V3},
  author={Jadhav, Dev},
  year={2024},
  url={https://github.com/DevJadhav/deepseek-from-scratch},
  license={Apache-2.0},
  note={Educational implementation of DeepSeek-V3 architecture including MLA, MoE, and MTP}
}
```

Also consider citing the original DeepSeek papers:

```bibtex
@article{deepseek_v3,
  title={DeepSeek-V3 Technical Report},
  author={DeepSeek-AI},
  journal={arXiv preprint arXiv:2412.19437},
  year={2024}
}

@article{deepseek_r1,
  title={DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning},
  author={DeepSeek-AI},
  journal={arXiv preprint arXiv:2501.12948},
  year={2025}
}
```

---

## 📚 References

- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [DeepSeek-R1 Technical Report](https://arxiv.org/abs/2501.12948)
- [Candle ML Framework](https://github.com/huggingface/candle)
- [MLX Framework](https://github.com/ml-explore/mlx)
- [Modal Cloud Platform](https://modal.com/)
- [Ray Framework](https://www.ray.io/)

---

## 📋 Changelog

See [CHANGELOG.md](CHANGELOG.md) for a detailed history of changes.

---

## ⭐ Acknowledgments

This project is for educational purposes, demonstrating the key architectural innovations in DeepSeek models. Special thanks to:

- DeepSeek AI for their open research and technical reports
- Hugging Face for the Candle framework
- Apple for the MLX framework
- The open-source ML community
