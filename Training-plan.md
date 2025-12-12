# DeepSeek Training Pipeline: Comprehensive Plan

> **Version**: 2.0
> **Generated**: December 2025
> **Repository**: DeepSeek-From-Scratch
> **Platform**: Modal (A100-80GB × 8)
> **Backends**: Rust+GPU vs PyTorch+GPU comparison
> **Budget**: $500 USD per backend ($1,000 total)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Repository Analysis](#2-repository-analysis)
3. [Cost Analysis & Budget Planning](#3-cost-analysis--budget-planning)
4. [Dataset Strategy](#4-dataset-strategy)
5. [Model Architecture Specifications](#5-model-architecture-specifications)
6. [Training Configuration Matrix](#6-training-configuration-matrix)
7. [Backend Comparison & Selection Guide](#7-backend-comparison--selection-guide)
8. [Ablation Study Roadmap](#8-ablation-study-roadmap)
9. [Implementation Timeline](#9-implementation-timeline)
10. [Resource Requirements](#10-resource-requirements)
11. [Risk Mitigation Strategies](#11-risk-mitigation-strategies)
12. [Appendices](#12-appendices)

---

## 1. Executive Summary

### Key Findings and Recommendations

| Aspect | Finding | Recommendation |
|--------|---------|----------------|
| **Platform** | Modal provides reliable A100-80GB infrastructure | Use Modal exclusively for all training runs |
| **Backend Comparison** | Rust+GPU vs PyTorch+GPU performance analysis | Run parallel training to compare throughput/efficiency |
| **Budget** | $500 per backend training run | Track costs in real-time with W&B and TensorBoard |
| **Hardware** | A100-80GB × 8 configuration | 640GB total VRAM, NVLink interconnect |
| **Data Mix** | 30% web, 30% code, 30% math, 5% papers, 5% scientific | Use temperature-scaled domain mixing |
| **Logging** | W&B (local) + TensorBoard for comprehensive tracking | Store artifacts locally with cloud sync option |

### Quick Reference Cost Table (Modal A100-80GB × 8)

| Model Size | Parameters | Training Tokens | GPU Hours (8×A100-80GB) | Est. Cost (Modal) | Time |
|------------|------------|-----------------|-------------------------|-------------------|------|
| **TINY** | 10M | 200M | 2 | $6 | 15 min |
| **256M** | 256M | 5.12B | 22 | $61 | 2.75 hrs |
| **512M** | 512M | 10.24B | 52 | $145 | 6.5 hrs |
| **1B** | 1B | 20B | 150 | $417 | 18.75 hrs |

*Costs calculated at Modal rate $2.78/hr per A100-80GB ($22.24/hr for 8×A100-80GB). Budget limit: $500 per backend.*

### Budget-Constrained Training Plan

| Backend | Budget | Max GPU Hours | Achievable Training |
|---------|--------|---------------|--------------------|
| **Rust+GPU** | $500 | ~180 hrs (8×A100) | Up to 1B model |
| **PyTorch+GPU** | $500 | ~180 hrs (8×A100) | Up to 1B model |
| **Total** | $1,000 | ~360 hrs | Comparative study |

*Both backends will train identical architectures for fair comparison.*

### Critical Decision Points

```
┌─────────────────────────────────────────────────────────────────┐
│                    PROJECT CONFIGURATION                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PLATFORM: Modal (Exclusive)                                    │
│  ─────────────────────────────────────────────────────────────  │
│  Hardware: A100-80GB × 8 (640GB total VRAM)                    │
│  Cost: $2.78/hr per GPU = $22.24/hr total                      │
│  Interconnect: NVLink for fast GPU-to-GPU communication        │
│                                                                 │
│  BACKEND COMPARISON STUDY                                       │
│  ─────────────────────────────────────────────────────────────  │
│  Backend A: Rust+GPU (Candle + CUDA)                           │
│  ├─ Budget: $500 USD                                           │
│  ├─ Focus: High-performance, low-level optimization            │
│  └─ Metrics: tok/s, memory efficiency, energy/token            │
│                                                                 │
│  Backend B: PyTorch+GPU (torch + CUDA)                         │
│  ├─ Budget: $500 USD                                           │
│  ├─ Focus: Flexibility, ecosystem integration                  │
│  └─ Metrics: tok/s, memory efficiency, energy/token            │
│                                                                 │
│  LOGGING & MONITORING                                           │
│  ─────────────────────────────────────────────────────────────  │
│  W&B: Local mode (wandb local) for experiment tracking         │
│  TensorBoard: Real-time training visualization                 │
│  Cost Tracker: $500 budget alerts per backend                  │
│                                                                 │
│  TRAINING TARGETS (within $500/backend)                        │
│  ─────────────────────────────────────────────────────────────  │
│  TINY (10M): Validation runs, ~$6 each                         │
│  256M: Architecture comparison, ~$61 each                      │
│  512M: Scaling behavior study, ~$145 each                      │
│  1B: Maximum within budget, ~$417 each                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Repository Analysis

### 2.1 Directory Structure Overview

```
DeepSeek-From-Scratch/
├── src/deepseek/                    # Main Python implementation (77,681 LOC)
│   ├── pipeline/                    # Ray-based orchestration layer
│   │   ├── config.py               # Unified configuration (1000+ LOC)
│   │   ├── workflow.py             # Training workflow orchestration
│   │   ├── data_ingestion.py       # Streaming data pipeline
│   │   ├── framework_selector.py   # Backend auto-selection (918 LOC)
│   │   ├── stages/                 # Pipeline stages (data_prep, pretrain, sft, grpo, distill, export)
│   │   ├── runners/                # Backend runners (pytorch, mlx, rust, modal)
│   │   └── utils/
│   │       └── data_downloader.py  # Dataset acquisition utility
│   │
│   ├── torch/                       # PyTorch implementations
│   │   ├── model/                  # Transformer, MLA, MoE, MTP, attention variants
│   │   ├── training/               # Training loops, FSDP, DualPipe, optimization
│   │   ├── utils/                  # Device, precision, benchmark utilities
│   │   └── kernels/                # Triton/CUDA custom kernels
│   │
│   ├── mlx/                         # Apple MLX implementations
│   │   ├── ane/                    # Apple Neural Engine support
│   │   ├── quantization.py         # MLX quantization
│   │   └── [model components]      # MLA, MoE, MTP mirrors
│   │
│   ├── cloud/modal/                 # Modal cloud training
│   │   ├── distributed_trainer.py  # Multi-GPU training
│   │   ├── ray_cluster.py          # Ray cluster management
│   │   ├── config.py               # Modal configuration
│   │   └── gpu_runner.py           # GPU orchestration
│   │
│   └── common/                      # Shared utilities
│       ├── storage.py              # S3/GCS/local abstraction
│       └── tracking/               # W&B, profiler integration
│
├── rust-src/                        # Rust/Candle high-performance backend
│   ├── src/
│   │   ├── model/                  # Transformer implementation
│   │   ├── distributed/            # NCCL backend
│   │   └── ablation/               # Paper experiments (A1-A6)
│   └── Cargo.toml                  # Candle + PyO3 dependencies
│
├── scripts/                         # Utility scripts
│   ├── ablation/                   # Ablation experiment scripts
│   ├── data-downloader/            # FineWeb download utilities
│   ├── train_tiny.py               # Single-machine training
│   ├── benchmark.py                # Performance benchmarking
│   └── evaluate.py                 # Model evaluation suite
│
├── monitoring/                      # Cost & performance tracking
│   ├── cost_tracker.py             # GPU cost calculation (806 LOC)
│   └── dashboard.py                # Rich terminal UI (675 LOC)
│
├── config/                          # Configuration files
│   ├── tiny_mlx_config.py          # Programmatic configs
│   └── hydra/                      # Hydra YAML configs
│
├── tests/                           # Comprehensive test suite
│   ├── pipeline/                   # Pipeline tests
│   ├── torch/                      # PyTorch tests
│   ├── mlx/                        # MLX tests
│   └── integration/                # E2E tests
│
└── docs/                            # Documentation
    └── 10-training-infrastructure.md
```

### 2.2 Key Components Map

| Component | Location | Purpose | LOC |
|-----------|----------|---------|-----|
| **Model Architecture** | `src/deepseek/torch/model/` | Transformer + MLA + MoE + MTP | 3,500+ |
| **Training Infrastructure** | `src/deepseek/torch/training/` | FSDP, DualPipe, optimization | 5,000+ |
| **Pipeline Orchestration** | `src/deepseek/pipeline/` | Ray-based workflow | 4,000+ |
| **Cost Tracking** | `monitoring/` | Budget management, alerts | 1,500+ |
| **Ablation Framework** | `scripts/ablation/` | Experiment infrastructure | 2,500+ |
| **Rust Backend** | `rust-src/` | High-performance inference | 3,000+ |
| **MLX Backend** | `src/deepseek/mlx/` | Apple Silicon support | 4,000+ |

### 2.3 Deprecated/Misaligned Components

| Component | Status | Action |
|-----------|--------|--------|
| Ray Workflows API | Replaced with Ray Tasks | No action needed (already migrated) |
| `backend` parameter in stages | Marked deprecated | Consider removal in next refactor |
| Ray Train API fallbacks | Handled with try/except | Compatible with multiple Ray versions |

**No orphaned model download scripts found.** All data utilities are actively integrated.

---

## 3. Cost Analysis & Budget Planning

### 3.1 Cost Infrastructure (Modal A100-80GB × 8)

The repository includes a comprehensive cost tracking system in `monitoring/cost_tracker.py`:

```python
# GPU Pricing Database (Modal - A100-80GB × 8 Configuration)
MODAL_PRICING = {
    "A100_80GB": {
        "price_per_hour": 2.78,      # Per GPU
        "gpus": 8,                    # Cluster size
        "total_per_hour": 22.24,     # 8 × $2.78
        "total_vram_gb": 640,        # 8 × 80GB
        "tdp_watts": 400,            # Per GPU
    }
}

# Backend Budget Configuration
BACKEND_BUDGETS = {
    "rust_gpu": {
        "budget_limit": 500.0,       # USD
        "max_gpu_hours": 22.5,       # $500 / $22.24/hr
        "alerts": [250, 375, 450, 475],  # 50%, 75%, 90%, 95%
    },
    "pytorch_gpu": {
        "budget_limit": 500.0,       # USD
        "max_gpu_hours": 22.5,       # $500 / $22.24/hr
        "alerts": [250, 375, 450, 475],  # 50%, 75%, 90%, 95%
    },
}
```

**Features:**
- Session-based tracking per backend (Rust+GPU vs PyTorch+GPU)
- Budget threshold alerts at $250, $375, $450, $475 per backend
- JSON persistence for recovery
- Real-time $/token tracking with backend comparison
- Energy efficiency metrics (J/token)
- W&B local integration for experiment tracking
- TensorBoard logging for real-time visualization

### 3.2 Modal GPU Configuration (Exclusive Platform)

| Configuration | GPUs | VRAM/GPU | Total VRAM | $/hr/GPU | Total $/hr |
|---------------|------|----------|------------|----------|------------|
| **A100-80GB × 8** | 8 | 80GB | 640GB | $2.78 | $22.24 |

**Why Modal Exclusively:**
- Consistent pricing and availability
- Native Ray cluster support for distributed training
- Integrated volume storage for checkpoints
- Built-in secrets management
- Serverless scaling (pay only for compute used)
- NVLink interconnect for fast GPU-to-GPU communication

### 3.2.1 Budget Allocation (Total: $1,000)

| Backend | Budget | GPU Hours (8×A100) | Max Training Time | Alert Thresholds |
|---------|--------|--------------------|--------------------|------------------|
| **Rust+GPU** | $500 | 22.5 hrs | ~22.5 hrs | $250, $375, $450, $475 |
| **PyTorch+GPU** | $500 | 22.5 hrs | ~22.5 hrs | $250, $375, $450, $475 |

**Note:** All training runs use Modal exclusively. No other cloud providers are used in this project.

### 3.3 Training Cost Calculations (Modal A100-80GB × 8)

#### Methodology

Using Chinchilla scaling laws with Modal A100-80GB × 8 cluster:
- **Compute-optimal tokens** ≈ 20 × parameters
- **GPU hours** = (6 × params × tokens) / (FLOPS × utilization)
- **A100-80GB FLOPS**: 312 TFLOPS (FP16 Tensor Cores)
- **Utilization**: 50% (optimized for 8-GPU distributed training)
- **Modal Rate**: $22.24/hr for 8×A100-80GB cluster

```
Cluster_hours = (6 × P × T) / (8 × 312e12 × 0.50 × 3600)
              ≈ (6 × P × T) / (4.49e18)
              ≈ 1.34e-18 × P × T
```

#### Budget-Constrained Model Training ($500 per backend)

| Model | Params | Tokens | FLOPs | Cluster-hrs | Cost (Modal) | Fits Budget? |
|-------|--------|--------|-------|-------------|--------------|---------------|
| TINY | 1×10⁷ | 2×10⁸ | 1.2×10¹⁶ | 0.25 | **$6** | ✅ Yes |
| 256M | 2.56×10⁸ | 5.12×10⁹ | 7.86×10¹⁸ | 2.75 | **$61** | ✅ Yes |
| 512M | 5.12×10⁸ | 1.02×10¹⁰ | 3.14×10¹⁹ | 6.5 | **$145** | ✅ Yes |
| 1B | 1×10⁹ | 2×10¹⁰ | 1.2×10²⁰ | 18.75 | **$417** | ✅ Yes |
| 3B | 3×10⁹ | 6×10¹⁰ | 1.08×10²¹ | 84 | **$1,870** | ❌ Over Budget |

**Budget Scenario: $500 per backend ($1,000 total)**
- Can train: TINY + 256M + 512M models on both backends
- Or: 1B model on both backends (best use of budget)
- Recommended: Train 1B on both Rust+GPU and PyTorch+GPU for comparison

#### Recommended Training Plan (Within $500/backend)

```
┌─────────────────────────────────────────────────────────────────┐
│          TRAINING PLAN PER BACKEND ($500 each)               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  OPTION A: Progressive Training (Recommended)                   │
│  ─────────────────────────────────────────────────────────────  │
│  1. TINY (10M) validation:        $6   → Cumulative: $6        │
│  2. 256M architecture test:       $61  → Cumulative: $67       │
│  3. 512M scaling study:           $145 → Cumulative: $212      │
│  4. Ablation experiments:         $88  → Cumulative: $300      │
│  5. Buffer for reruns:            $200 → Cumulative: $500      │
│                                                                 │
│  OPTION B: Single Large Run (Maximum Scale)                     │
│  ─────────────────────────────────────────────────────────────  │
│  1. TINY validation:              $6   → Cumulative: $6        │
│  2. 1B full training:             $417 → Cumulative: $423      │
│  3. Buffer for evaluation:        $77  → Cumulative: $500      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.4 Cost Breakdown Components (Per Backend - $500 Budget)

```
┌─────────────────────────────────────────────────────────────────┐
│         COST BREAKDOWN PER BACKEND ($500 Budget)                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  RUST+GPU Backend ($500)                                        │
│  ────────────────────────────────────────────────────────────   │
│  Compute (Modal A100-80GB×8)    $475  (95%)                    │
│  ├─ TINY validation               $6   (1.2%)                  │
│  ├─ 256M training                $61  (12.2%)                  │
│  ├─ 512M training               $145  (29%)                    │
│  ├─ Ablation experiments         $88  (17.6%)                  │
│  └─ Buffer/reruns               $175  (35%)                    │
│                                                                 │
│  Storage (Modal Volumes)         $15  (3%)                     │
│  Logs (W&B local + TB)           $10  (2%)                     │
│                                                                 │
│  PYTORCH+GPU Backend ($500)                                     │
│  ────────────────────────────────────────────────────────────   │
│  Compute (Modal A100-80GB×8)    $475  (95%)                    │
│  ├─ TINY validation               $6   (1.2%)                  │
│  ├─ 256M training                $61  (12.2%)                  │
│  ├─ 512M training               $145  (29%)                    │
│  ├─ Ablation experiments         $88  (17.6%)                  │
│  └─ Buffer/reruns               $175  (35%)                    │
│                                                                 │
│  Storage (Modal Volumes)         $15  (3%)                     │
│  Logs (W&B local + TB)           $10  (2%)                     │
│                                                                 │
│  COMBINED TOTAL               $1,000                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.5 Cost Optimization Strategies (Modal-Specific)

| Strategy | Savings | Implementation |
|----------|---------|----------------|
| **Serverless Scaling** | Pay only for compute | Modal auto-scales to 0 when idle |
| **Mixed Precision (BF16)** | 50% memory | Enable in config |
| **Gradient Checkpointing** | 40% memory | Trade compute for memory |
| **Batch Size Optimization** | 10-20% | Binary search for optimal |
| **Early Stopping** | Variable | Stop if loss plateaus |
| **Checkpoint Compression** | 20-30% storage | Use safetensors format |

### 3.6 Amortized Cost Per Token (Within Budget)

| Model Size | Total Tokens | Cost (Modal) | Cost/Million Tokens |
|------------|--------------|--------------|---------------------|
| TINY | 200M | $6 | $0.030 |
| 256M | 5.12B | $61 | $0.012 |
| 512M | 10.24B | $145 | $0.014 |
| 1B | 20B | $417 | $0.021 |

*Note: All models within $500 per backend budget.*

---

## 4. Dataset Strategy

### 4.1 Dataset Configuration

The repository integrates five curated datasets via `data_downloader.py`:

```python
DOMAIN_DATASETS = {
    "web": {
        "hf_path": "HuggingFaceFW/fineweb-edu",
        "subset": "sample-10BT",
        "text_field": "text",
        "weight": 0.30,  # 30% of training data
    },
    "code": {
        "hf_path": "mlfoundations-dev/stackoverflow",  # To integrate
        # Currently: "flytech/python-codes-25k"
        "text_field": "output",
        "weight": 0.30,  # 30% of training data
    },
    "math": {
        "hf_path": "HuggingFaceTB/finemath",
        "subset": "finemath-3plus",
        "text_field": "text",
        "weight": 0.30,  # 30% of training data
    },
    "books": {
        "hf_path": "CShorten/ML-ArXiv-Papers",
        "text_field": "abstract",
        "weight": 0.05,  # 5% of training data
    },
    "scientific": {
        "hf_path": "jamescalam/ai-arxiv",
        "text_field": "chunk",
        "weight": 0.05,  # 5% of training data
    },
}
```

### 4.2 Dataset Sizes and Token Counts

| Dataset | Raw Size | Est. Tokens | Quality Score | Notes |
|---------|----------|-------------|---------------|-------|
| FineWeb-Edu (10BT) | ~50GB | 10B | High | Educational filtering applied |
| FineWeb-Edu (Full) | ~15TB | 1.5T | High | Full corpus for large models |
| StackOverflow | ~60GB | 30B | Medium-High | Code + explanations |
| FineMath | ~10GB | 5B | High | Math-focused filtering |
| ML-ArXiv-Papers | ~2GB | 500M | High | Research paper abstracts |
| AI-ArXiv | ~5GB | 2B | High | AI research chunks |

### 4.3 Recommended Data Mixing Ratios by Model Size

```
┌─────────────────────────────────────────────────────────────────┐
│              DATA MIXING RATIOS BY MODEL SCALE                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Small (256M-1B):                                              │
│  ├─ Web:        30%  (General language understanding)          │
│  ├─ Code:       30%  (Basic programming patterns)              │
│  ├─ Math:        30%  (Numerical reasoning)                     │
│  ├─ Papers:      5%  (Academic vocabulary)                     │
│  └─ Scientific:  5%  (Domain exposure)                         │
│                                                                 │
│  Medium (3B-8B):                                                │
│  ├─ Web:        60%  (Balanced foundation)                     │
│  ├─ Code:       20%  (Strong coding capability)                │
│  ├─ Math:       10%  (Mathematical reasoning)                  │
│  ├─ Papers:      5%  (Research understanding)                  │
│  └─ Scientific:  5%  (Technical depth)                         │
│                                                                 │
│  Large (12B-24B):                                               │
│  ├─ Web:        55%  (Diverse knowledge)                       │
│  ├─ Code:       22%  (Advanced programming)                    │
│  ├─ Math:       12%  (Complex reasoning)                       │
│  ├─ Papers:      6%  (Academic depth)                          │
│  └─ Scientific:  5%  (Specialized knowledge)                   │
│                                                                 │
│  Extra Large (48B-100B):                                        │
│  ├─ Web:        50%  (World knowledge)                         │
│  ├─ Code:       25%  (Expert-level coding)                     │
│  ├─ Math:       12%  (Advanced mathematics)                    │
│  ├─ Papers:      7%  (Research capability)                     │
│  └─ Scientific:  6%  (Domain expertise)                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 Preprocessing Pipeline

```python
# From data_ingestion.py
DataIngestionPipeline:
    1. StreamingDataPipeline
       ├─ Load datasets in streaming mode (no full dataset in memory)
       ├─ Lazy tokenizer loading (deepseek-ai/deepseek-llm-7b-base)
       └─ Worker-aware sharding for distributed training

    2. DeterministicShuffler
       ├─ Hierarchical RNG (Master → Per-Worker → Per-Epoch)
       ├─ Streaming shuffle with 10,000 buffer
       └─ State checkpointing for resume capability

    3. DataMixer
       ├─ Temperature-scaled domain mixing: P(d_i) = w_i^τ / Σ w_j^τ
       ├─ τ = 1.0 (default): Use original weights
       ├─ τ > 1.0: Softer distribution (reduces dominance)
       └─ τ < 1.0: Harder distribution (increases contrast)

    4. TokenLevelBatcher
       ├─ Token budget per batch: 16,384 tokens
       ├─ Packing mode: Multiple sequences in one array
       ├─ Padded mode: Traditional 2D with padding
       └─ Document boundary tracking (no cross-doc attention)

    5. DynamicPadder
       ├─ Sequence length buckets: [64, 128, 256, 512, 1024, 2048, 4096]
       ├─ Pad only to bucket max (not global max)
       └─ Target efficiency: 85% (real tokens / allocated)
```

### 4.5 Curriculum Learning Configuration

```python
CurriculumScheduler:
    start_seq_len: 512      # Begin with shorter sequences
    end_seq_len: 4096       # Grow to full context
    seq_curriculum_steps: 10000  # Steps for sequence growth
    difficulty_warmup_steps: 5000  # Easy → All difficulties

    # Progression:
    # Step 0:     512 tokens, easy samples only
    # Step 2500:  1024 tokens, 50% difficulty mix
    # Step 5000:  2048 tokens, all difficulties
    # Step 10000: 4096 tokens, full curriculum
```

### 4.6 Data Download Commands

```bash
# Download all datasets
uv run python src/deepseek/pipeline/utils/data_downloader.py \
    --output-dir ./data \
    --max-samples 100000 \
    --domains web code math books scientific

# Download FineWeb-Edu specifically
uv run python scripts/data-downloader/download_fineweb_edu.py \
    --output-dir ./data/fineweb \
    --subset sample-10BT

# Tokenize downloaded data
uv run python scripts/data-downloader/tokenize_fineweb.py \
    --input-dir ./data/fineweb \
    --output-dir ./data/tokenized \
    --tokenizer deepseek-ai/deepseek-llm-7b-base
```

---

## 5. Model Architecture Specifications

### 5.1 Model Size Configurations

| Config | d_model | n_heads | n_layers | FFN | Vocab | Params |
|--------|---------|---------|----------|-----|-------|--------|
| **TINY** | 256 | 4 | 4 | 1024 | 32K | ~10M |
| **256M** | 768 | 12 | 12 | 3072 | 100K | 256M |
| **512M** | 1024 | 16 | 16 | 4096 | 100K | 512M |
| **1B** | 1536 | 16 | 24 | 6144 | 100K | 1B |
| **3B** | 2048 | 16 | 32 | 8192 | 100K | 3B |
| **5B** | 2560 | 20 | 36 | 10240 | 100K | 5B |
| **8B** | 4096 | 32 | 32 | 14336 | 100K | 8B |
| **12B** | 4096 | 32 | 48 | 14336 | 100K | 12B |
| **24B** | 5120 | 40 | 56 | 17920 | 100K | 24B |
| **48B** | 6144 | 48 | 64 | 21504 | 100K | 48B |
| **100B** | 8192 | 64 | 80 | 28672 | 100K | 100B |

### 5.2 DeepSeek Architecture Components

#### Multi-Head Latent Attention (MLA)

```python
# From src/deepseek/torch/model/mla.py
MLA Configuration:
    d_latent: 512           # Latent dimension for KV compression
    d_rope: 64              # RoPE dimension (decoupled)
    compression_ratio: 14x  # KV cache reduction

    # KV Cache Comparison (8B model, 4096 context):
    # Standard MHA: 8B × 32 heads × 2 (K,V) × 4096 × 128 = 8.6GB
    # MLA (14x):    8.6GB / 14 = 614MB
```

#### Mixture of Experts (MoE)

```python
# From src/deepseek/torch/model/moe.py
MoE Configuration:
    num_experts: 256        # Total routed experts
    num_shared_experts: 1   # Always-active shared expert
    top_k: 8                # Experts per token
    expert_groups: 8        # Hierarchical routing groups
    capacity_factor: 1.5    # Overflow buffer

    # Activation: 37B of 671B (5.5% active)
    # Load balancing: Auxiliary-loss-free (sigmoid gating)
```

#### Multi-Token Prediction (MTP)

```python
# From src/deepseek/torch/model/mtp.py
MTP Configuration:
    mtp_k: 3                # Predict 3 future tokens
    acceptance_rate: 85%    # Speculative decoding acceptance

    # Training overhead: 1.0x → 1.35x
    # Inference speedup: 1.5-2.5x
```

### 5.3 Architecture by Model Scale

```
┌─────────────────────────────────────────────────────────────────┐
│              ARCHITECTURE SCALING STRATEGY                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Small (256M-1B): Dense Transformer                            │
│  ├─ Standard MHA (no MLA compression needed)                   │
│  ├─ No MoE (dense FFN sufficient)                              │
│  ├─ MTP depth: 1 (single next-token prediction)                │
│  └─ Context: 2048-4096 tokens                                  │
│                                                                 │
│  Medium (3B-8B): Add MLA + Optional MoE                        │
│  ├─ MLA: Enable KV compression (d_latent=512)                  │
│  ├─ MoE: 8-16 experts, top-2 routing                          │
│  ├─ MTP depth: 2 (predict 2 tokens)                           │
│  └─ Context: 4096-8192 tokens                                  │
│                                                                 │
│  Large (12B-24B): Full MLA + MoE                               │
│  ├─ MLA: Full compression (d_latent=768)                       │
│  ├─ MoE: 64 experts, top-4 routing                            │
│  ├─ MTP depth: 2-3                                            │
│  └─ Context: 8192-32768 tokens                                 │
│                                                                 │
│  Extra Large (48B-100B): DeepSeek-V3 Style                     │
│  ├─ MLA: Maximum compression (d_latent=1024)                   │
│  ├─ MoE: 256 experts, top-8, hierarchical routing             │
│  ├─ MTP depth: 3                                              │
│  └─ Context: 32768-131072 tokens                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.4 Memory Requirements per Model

| Model | Params (FP16) | Optimizer (FP32) | Gradients | Activations* | Total (Training) |
|-------|---------------|------------------|-----------|--------------|------------------|
| 256M | 0.5 GB | 1.0 GB | 0.5 GB | 2 GB | **4 GB** |
| 1B | 2 GB | 4 GB | 2 GB | 8 GB | **16 GB** |
| 3B | 6 GB | 12 GB | 6 GB | 16 GB | **40 GB** |
| 8B | 16 GB | 32 GB | 16 GB | 32 GB | **96 GB** |
| 24B | 48 GB | 96 GB | 48 GB | 64 GB | **256 GB** |
| 100B | 200 GB | 400 GB | 200 GB | 200 GB | **1 TB** |

*Activations with gradient checkpointing (selective). Without checkpointing, multiply by 3-4x.

---

## 6. Training Configuration Matrix

### 6.1 Hyperparameter Configurations (Budget-Constrained Models)

| Parameter | TINY (10M) | 256M | 512M | 1B |
|-----------|------------|------|------|-----|
| **Learning Rate** | 5e-4 | 3e-4 | 3e-4 | 3e-4 |
| **Batch Size (tokens)** | 64K | 256K | 256K | 512K |
| **Warmup Steps** | 100 | 500 | 500 | 1000 |
| **Weight Decay** | 0.1 | 0.1 | 0.1 | 0.1 |
| **Gradient Clip** | 1.0 | 1.0 | 1.0 | 1.0 |
| **LR Schedule** | Cosine | Cosine | Cosine | Cosine |
| **β1, β2** | 0.9, 0.95 | 0.9, 0.95 | 0.9, 0.95 | 0.9, 0.95 |
| **Context Length** | 512 | 2048 | 2048 | 4096 |
| **Precision** | BF16 | BF16 | BF16 | BF16 |

### 6.2 Modal A100-80GB × 8 Configuration

```python
# Modal 8×A100-80GB Configuration (Project Standard)
modal_8gpu_config = {
    "gpu_type": "A100-80GB",
    "gpu_count": 8,
    "total_vram": 640,             # GB
    "interconnect": "NVLink",
    "cost_per_hour": 22.24,        # $2.78/GPU × 8
    "budget_per_backend": 500.0,   # USD
    "max_hours": 22.5,             # Hours within budget
}

# Parallelism Strategy (8×A100-80GB)
parallelism_modal = {
    "tensor_parallel": 2,          # Split within GPU pairs
    "pipeline_parallel": 2,        # 2 pipeline stages
    "data_parallel": 2,            # 2 replicas
    "expert_parallel": 1,          # No expert sharding
    "sequence_parallel": 1,        # No sequence sharding
    # Total: TP×PP×DP = 2×2×2 = 8 GPUs
}
```

### 6.3 Backend-Specific Configurations

```python
# Rust+GPU Backend Configuration
rust_gpu_config = {
    "backend": "candle_cuda",
    "precision": "bf16",
    "gradient_checkpointing": True,
    "compile": True,               # Use CUDA graph optimization
    "logging": {
        "wandb_mode": "offline",   # W&B local mode
        "tensorboard": True,
        "log_dir": "/vol/logs/rust_gpu",
    },
    "budget_limit": 500.0,
    "alert_thresholds": [250, 375, 450, 475],
}

# PyTorch+GPU Backend Configuration
pytorch_gpu_config = {
    "backend": "torch_cuda",
    "precision": "bf16",
    "use_fsdp": True,
    "gradient_checkpointing": True,
    "compile": True,               # torch.compile
    "logging": {
        "wandb_mode": "offline",   # W&B local mode
        "tensorboard": True,
        "log_dir": "/vol/logs/pytorch_gpu",
    },
    "budget_limit": 500.0,
    "alert_thresholds": [250, 375, 450, 475],
}
```

### 6.3 Checkpoint Strategies

```python
CheckpointConfig:
    # Frequency
    save_every_n_steps: 500        # Regular checkpoints
    keep_last_n: 3                 # Rolling window
    save_on_emergency: True        # On failure

    # Sharding (for large models)
    use_sharded_checkpoint: True   # For FSDP models
    checkpoint_format: "safetensors"  # Fast, safe format

    # Sizes (approximate)
    # 1B model: ~4 GB per checkpoint
    # 8B model: ~32 GB per checkpoint
    # 24B model: ~96 GB per checkpoint

    # Storage costs (S3): ~$0.023/GB/month
    # 8B with 3 checkpoints: 96 GB × $0.023 = $2.21/month
```

### 6.4 Training Stage Configurations

```python
# Stage 1: Pretraining
PretrainConfig:
    max_steps: 100000              # Adjust per model size
    eval_every_n_steps: 1000
    log_every_n_steps: 10
    optimizer: "adamw"
    scheduler: "cosine_with_warmup"
    use_amp: True                  # Automatic Mixed Precision
    gradient_accumulation: 4       # Effective batch scaling

# Stage 2: Supervised Fine-Tuning (SFT)
SFTConfig:
    max_steps: 5000
    learning_rate: 2e-5            # Lower than pretraining
    use_lora: True                 # LoRA for efficiency
    lora_r: 16
    lora_alpha: 32
    lora_dropout: 0.1
    target_modules: ["q_proj", "v_proj", "o_proj"]

# Stage 3: GRPO (Reinforcement Learning)
GRPOConfig:
    max_steps: 2000
    learning_rate: 1e-6            # Very low for stability
    kl_coefficient: 0.1
    reward_model: "path/to/reward_model"
    num_rollouts: 4
    ppo_epochs: 2
```

### 6.5 DeepSpeed Configuration Templates

```json
// ZeRO Stage 2 (Recommended for 1B-8B)
{
    "train_batch_size": 256,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": {"lr": 1.5e-4, "weight_decay": 0.1}
    },
    "fp16": {"enabled": true},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "none"},
        "contiguous_gradients": true,
        "overlap_comm": true
    }
}

// ZeRO Stage 3 (Required for 24B+)
{
    "train_batch_size": 512,
    "gradient_accumulation_steps": 8,
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu"},
        "offload_param": {"device": "cpu"},
        "overlap_comm": true,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8
    }
}
```

---

## 7. Backend Comparison: Rust+GPU vs PyTorch+GPU

### 7.1 Feature Comparison Matrix

| Feature | PyTorch+GPU | Rust+GPU (Candle) |
|---------|-------------|-------------------|
| **Training Support** | Full (FSDP, DDP) | Experimental (NCCL) |
| **torch.compile** | Yes | N/A (native CUDA graphs) |
| **Mixed Precision** | BF16/FP16 native | BF16/FP16/FP8 |
| **Memory Efficiency** | Good | Excellent |
| **Startup Time** | 2-3s | 3-5s |
| **Inference Speed** | 50-100 tok/s | 100-200 tok/s |
| **Ecosystem** | Mature | Growing |
| **Debugging** | Excellent | Limited |
| **Code Maturity** | Production | Advanced |
| **A100 Optimization** | Generic | Native CUDA |

### 7.2 Performance Benchmarks (Modal A100-80GB × 8)

| Metric | PyTorch+GPU | Rust+GPU | Winner |
|--------|-------------|----------|--------|
| **Training tok/s (256M)** | ~4000 | ~3800 | PyTorch |
| **Training tok/s (512M)** | ~2500 | ~2600 | Rust |
| **Training tok/s (1B)** | ~1500 | ~1600 | Rust |
| **Memory/Param (BF16)** | 2.0 GB | 1.8 GB | Rust |
| **GPU Utilization** | 45% | 52% | Rust |
| **Startup Latency** | 2.5s | 4s | PyTorch |
| **Checkpoint Save** | 15s | 12s | Rust |

*Benchmarks are estimates. Actual comparison will be performed during training.*

### 7.3 Backend Selection Decision Tree

```
┌─────────────────────────────────────────────────────────────────┐
### 7.3 Backend Comparison Study Plan

```
┌─────────────────────────────────────────────────────────────────┐
│         BACKEND COMPARISON STUDY (Rust+GPU vs PyTorch+GPU)      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  OBJECTIVE: Compare training performance on identical tasks     │
│  PLATFORM: Modal (A100-80GB × 8)                               │
│  BUDGET: $500 per backend ($1,000 total)                       │
│                                                                 │
│  COMPARISON METRICS:                                            │
│  ─────────────────────────────────────────────────────────────  │
│  • Throughput (tokens/second)                                   │
│  • Memory efficiency (GB/billion params)                        │
│  • GPU utilization (%)                                          │
│  • Energy efficiency (J/token)                                  │
│  • Training stability (loss variance)                           │
│  • Checkpoint save/load time                                    │
│  • Model quality (validation loss)                              │
│                                                                 │
│  STUDY PHASES:                                                  │
│  ─────────────────────────────────────────────────────────────  │
│  Phase 1: TINY (10M) - Validation ($6 each)                    │
│  ├─ Verify both backends work correctly                        │
│  ├─ Establish baseline metrics                                 │
│  └─ Test logging (W&B local + TensorBoard)                     │
│                                                                 │
│  Phase 2: 256M - Architecture Test ($61 each)                  │
│  ├─ Compare throughput at larger scale                         │
│  ├─ Memory usage comparison                                    │
│  └─ Training stability analysis                                │
│                                                                 │
│  Phase 3: 512M - Scaling Study ($145 each)                     │
│  ├─ Scaling efficiency comparison                              │
│  ├─ Multi-GPU communication overhead                           │
│  └─ Final model quality comparison                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.4 Logging & Monitoring Configuration

```python
# W&B Local Mode Configuration
wandb_config = {
    "mode": "offline",              # Run W&B locally
    "project": "deepseek-training",
    "entity": "local",
    "dir": "/vol/wandb",            # Modal volume for persistence
    "tags": ["modal", "a100-80gb", "comparison"],
    "config": {
        "backend": "rust_gpu | pytorch_gpu",
        "budget_limit": 500.0,
        "gpu_type": "A100-80GB",
        "gpu_count": 8,
    },
}

# TensorBoard Configuration
tensorboard_config = {
    "log_dir": "/vol/tensorboard/{backend}/{run_id}",
    "flush_secs": 30,
    "max_queue": 100,
    "write_images": False,          # Save storage
    "write_histograms": True,
    "profile_batch": "10,20",       # Profile batches 10-20
}

# Combined Logging Setup
logging_config = {
    "wandb": wandb_config,
    "tensorboard": tensorboard_config,
    "log_every_n_steps": 10,
    "eval_every_n_steps": 100,
    "checkpoint_every_n_steps": 500,
    "artifact_dir": "/vol/artifacts/{backend}",
}
```

---

## 8. Ablation Study Roadmap

### 8.1 Current Ablation Infrastructure

Located in `scripts/ablation/`:

| Script | Purpose | Status |
|--------|---------|--------|
| `ablation_utils.py` | Core infrastructure (AblationConfig, AblationRunner) | Complete |
| `run_attention_ablation.py` | MLA vs GQA vs MHA comparison | Complete |
| `run_expert_ablation.py` | 8 vs 64 vs 256 experts | Complete |
| `run_balancing_ablation.py` | Aux-loss-free vs aux-loss variants | Complete |
| `run_mtp_ablation.py` | MTP depth D=0,1,2,3 | Complete |
| `run_precision_ablation.py` | FP32/FP16/BF16/FP8 | Mocked (needs real impl) |
| `run_rope_ablation.py` | RoPE variants (standard, NTK, YaRN) | Complete |
| `run_all_ablations.py` | Master orchestration script | Complete |

### 8.2 Gap Analysis

```
┌─────────────────────────────────────────────────────────────────┐
│                    ABLATION GAP ANALYSIS                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  EXISTING (✅)           MISSING (❌)           PRIORITY        │
│  ─────────────────────────────────────────────────────────────  │
│  Attention variants      Backend comparison       HIGH          │
│  Expert count            Model scaling           CRITICAL       │
│  Load balancing          Dataset mixture         CRITICAL       │
│  MTP depth               Learning rate schedules MEDIUM         │
│  Precision (mocked)      Precision (real)        MEDIUM         │
│  RoPE variants           Expert routing          MEDIUM         │
│                          Batch size effects      LOW            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.3 Ablation Study Task Roadmap

#### Task Checklist (Integrated with Training)

| ID | Task | Backend | Budget Est. | Status | Dependencies |
|----|------|---------|-------------|--------|--------------|
| **A1** | Setup W&B local + TensorBoard | Both | $0 | TODO | None |
| **A2** | TINY validation (Rust+GPU) | Rust | $6 | TODO | A1 |
| **A3** | TINY validation (PyTorch+GPU) | PyTorch | $6 | TODO | A1 |
| **A4** | Backend throughput comparison (TINY) | Both | $0 | TODO | A2, A3 |
| **A5** | 256M attention ablation (Rust+GPU) | Rust | $20 | TODO | A4 |
| **A6** | 256M attention ablation (PyTorch+GPU) | PyTorch | $20 | TODO | A4 |
| **A7** | Compare attention results | Both | $0 | TODO | A5, A6 |
| **A8** | 256M MTP depth ablation | PyTorch | $25 | TODO | A6 |
| **A9** | 256M precision ablation (BF16 vs FP16) | Both | $15 | TODO | A4 |
| **A10** | 512M scaling validation | Both | $145×2 | TODO | A7 |
| **A11** | Dataset mixture mini-study | PyTorch | $50 | TODO | A6 |
| **A12** | Final comparison report | N/A | $0 | TODO | All |

**Total Estimated Ablation Budget:** ~$150 per backend (included in $500 total)

#### Detailed Task Breakdown

```
┌─────────────────────────────────────────────────────────────────┐
│              ABLATION STUDY TASK ROADMAP                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PHASE 1: Infrastructure Setup (Day 1)                         │
│  ─────────────────────────────────────────────────────────────  │
│  □ A1.1 Configure W&B local mode on Modal volume               │
│  □ A1.2 Setup TensorBoard logging directory                    │
│  □ A1.3 Create budget tracking hooks for both backends         │
│  □ A1.4 Verify artifact storage on Modal volumes               │
│                                                                 │
│  PHASE 2: Validation Runs (Day 2-3)                            │
│  ─────────────────────────────────────────────────────────────  │
│  □ A2.1 Run TINY model on Rust+GPU backend                     │
│  □ A2.2 Verify loss convergence and logging                    │
│  □ A3.1 Run TINY model on PyTorch+GPU backend                  │
│  □ A3.2 Compare training curves in TensorBoard                 │
│  □ A4.1 Generate throughput comparison report                  │
│                                                                 │
│  PHASE 3: 256M Ablations (Day 4-7)                             │
│  ─────────────────────────────────────────────────────────────  │
│  □ A5.1 MLA vs GQA vs MHA on Rust+GPU                         │
│  □ A6.1 MLA vs GQA vs MHA on PyTorch+GPU                      │
│  □ A7.1 Statistical analysis of attention results              │
│  □ A8.1 MTP depth D=0,1,2 comparison                          │
│  □ A9.1 BF16 vs FP16 precision comparison                     │
│                                                                 │
│  PHASE 4: Scaling & Data (Day 8-14)                            │
│  ─────────────────────────────────────────────────────────────  │
│  □ A10.1 Train 512M on Rust+GPU (full run)                    │
│  □ A10.2 Train 512M on PyTorch+GPU (full run)                 │
│  □ A11.1 Dataset mixture: web-only baseline                   │
│  □ A11.2 Dataset mixture: web+code+math                       │
│                                                                 │
│  PHASE 5: Analysis & Reporting (Day 15)                        │
│  ─────────────────────────────────────────────────────────────  │
│  □ A12.1 Generate comparison plots                             │
│  □ A12.2 Statistical significance tests                        │
│  □ A12.3 Write final comparison report                         │
│  □ A12.4 Archive all W&B runs and TB logs                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.4 Backend Comparison Ablation (`run_backend_comparison.py`)

```python
# Proposed structure for Rust+GPU vs PyTorch+GPU comparison
BackendComparisonAblation:
    backends: ["rust_gpu", "pytorch_gpu"]
    platform: "modal"
    gpu_config: "A100-80GB × 8"
    
    metrics: [
        "throughput_tok_s",
        "memory_peak_gb",
        "gpu_utilization_pct",
        "energy_j_per_tok",
        "checkpoint_save_time_s",
        "loss_at_1k_steps",
    ]
    
    configurations:
        model_sizes: ["tiny", "256M", "512M"]
        batch_sizes: [64, 128, 256]
        seq_lengths: [512, 1024, 2048]
    
    budget_per_backend: 500.0
    seeds: [42, 123, 456]
    
    logging:
        wandb_mode: "offline"
        tensorboard: True
        artifact_dir: "/vol/ablation_artifacts"
```

### 8.5 Ablation Execution Commands (Modal)

```bash
# Phase 1: Infrastructure validation
uv run modal run src/deepseek/cloud/modal/ray_cluster.py::setup_logging \
    --wandb-mode offline \
    --tensorboard-dir /vol/tensorboard

# Phase 2: TINY validation runs
uv run modal run src/deepseek/cloud/modal/ray_cluster.py::run_rust \
    --model-size tiny \
    --max-steps 1000 \
    --budget-limit 6.0 \
    --output-dir /vol/ablation/rust_tiny

uv run modal run src/deepseek/cloud/modal/ray_cluster.py::run_pytorch \
    --model-size tiny \
    --max-steps 1000 \
    --budget-limit 6.0 \
    --output-dir /vol/ablation/pytorch_tiny

# Phase 3: 256M ablations
uv run python scripts/ablation/run_attention_ablation.py \
    --backend rust_gpu \
    --model-size 256M \
    --platform modal \
    --budget-limit 20.0

uv run python scripts/ablation/run_attention_ablation.py \
    --backend pytorch_gpu \
    --model-size 256M \
    --platform modal \
    --budget-limit 20.0

# Phase 4: 512M scaling
uv run modal run src/deepseek/cloud/modal/ray_cluster.py::run_rust \
    --model-size 512M \
    --max-steps 5000 \
    --budget-limit 145.0

uv run modal run src/deepseek/cloud/modal/ray_cluster.py::run_pytorch \
    --model-size 512M \
    --max-steps 5000 \
    --budget-limit 145.0
```

### 8.6 Ablation Result Analysis

The ablation framework provides:

```python
# Statistical analysis
from scripts.ablation.ablation_utils import statistical_analysis, generate_latex_table

results = statistical_analysis(ablation_results)
# Returns: t-tests, Cohen's d effect sizes, 95% confidence intervals

# Publication-ready output
latex = generate_latex_table(results, caption="Ablation Results")
```

---

## 9. Implementation Timeline

### 9.1 Gantt Chart (ASCII)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        IMPLEMENTATION TIMELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE 1: INFRASTRUCTURE (Weeks 1-2)                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ W1: Data pipeline setup      ████████                                │  │
│  │ W1: Cost tracking validation ████████                                │  │
│  │ W2: Backend testing          ████████████████                        │  │
│  │ W2: Ablation framework       ████████████████                        │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  PHASE 2: TRAINING & COMPARISON (Days 5-14)                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ D5-6: TINY validation         ████████                                │  │
│  │ D6-8: 256M Rust+GPU                  ████████████                    │  │
│  │ D6-8: 256M PyTorch+GPU                       ████████████            │  │
│  │ D9-12: 512M both backends                            ████████████████│  │
│  │ D13-14: Ablation studies                                     ████████│  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  PHASE 3: ANALYSIS & REPORTING (Days 15-16)                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ D15: Comparison analysis     ████████████████████████                │  │
│  │ D16: Final report                    ████████████████████████████████│  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Phase Details (Budget: $500 per Backend)

#### Phase 1: Infrastructure Setup (Days 1-4)

| Task | Duration | Dependencies | Deliverables |
|------|----------|--------------|--------------|
| Data pipeline validation | 1 day | None | Downloaded datasets, tokenization verified |
| Cost tracking setup ($500/backend) | 0.5 day | None | Budget alerts at $250, $375, $450, $475 |
| W&B local + TensorBoard setup | 0.5 day | None | Logging verified on Modal volumes |
| Backend testing (Rust+GPU) | 1 day | Logging | Rust backend verified on TINY |
| Backend testing (PyTorch+GPU) | 1 day | Logging | PyTorch backend verified on TINY |

**Commands:**
```bash
# Day 1: Data pipeline
uv run python src/deepseek/pipeline/utils/data_downloader.py --domains all
uv run pytest tests/monitoring/test_cost_tracker.py -v

# Day 2: Cost & logging setup
uv run modal run src/deepseek/cloud/modal/app.py::setup_budget_tracking \
    --rust-budget 500 --pytorch-budget 500
uv run modal run src/deepseek/cloud/modal/app.py::setup_wandb_local
uv run modal run src/deepseek/cloud/modal/app.py::setup_tensorboard

# Days 3-4: Backend validation
uv run modal run src/deepseek/cloud/modal/ray_cluster.py::run_rust --model-size tiny
uv run modal run src/deepseek/cloud/modal/ray_cluster.py::run_pytorch --model-size tiny
```

#### Phase 2: Training & Comparison (Days 5-14)

| Model | Backend | Days | Est. Cost | Checkpoints |
|-------|---------|------|-----------|-------------|
| TINY | Both | D5 | $6×2 = $12 | 2 |
| 256M | Rust+GPU | D6-8 | $61 | 4 |
| 256M | PyTorch+GPU | D6-8 | $61 | 4 |
| 512M | Rust+GPU | D9-12 | $145 | 6 |
| 512M | PyTorch+GPU | D9-12 | $145 | 6 |
| Ablations | PyTorch | D13-14 | $50 | N/A |

**Commands:**
```bash
# Days 5: TINY validation (both backends in parallel)
uv run modal run src/deepseek/cloud/modal/ray_cluster.py::run_rust \
    --model-size tiny --max-steps 1000 --budget-limit 6.0

uv run modal run src/deepseek/cloud/modal/ray_cluster.py::run_pytorch \
    --model-size tiny --max-steps 1000 --budget-limit 6.0

# Days 6-8: 256M training (both backends)
uv run modal run src/deepseek/cloud/modal/ray_cluster.py::run_rust \
    --model-size 256M --max-steps 5000 --budget-limit 61.0 \
    --checkpoint-dir /vol/checkpoints/rust/256M

uv run modal run src/deepseek/cloud/modal/ray_cluster.py::run_pytorch \
    --model-size 256M --max-steps 5000 --budget-limit 61.0 \
    --checkpoint-dir /vol/checkpoints/pytorch/256M

# Days 9-12: 512M training (both backends)
uv run modal run src/deepseek/cloud/modal/ray_cluster.py::run_rust \
    --model-size 512M --max-steps 8000 --budget-limit 145.0

uv run modal run src/deepseek/cloud/modal/ray_cluster.py::run_pytorch \
    --model-size 512M --max-steps 8000 --budget-limit 145.0 --use-fsdp
```

# 5B training
uv run modal run src/deepseek/cloud/modal/distributed_trainer.py::train_distributed \
    --model-size 5B \
    --max-steps 80000 \
    --deepspeed-zero-stage 2
```

---

## 10. Resource Requirements

### 10.1 Compute Requirements by Model Scale

| Scale | GPUs | GPU Type | VRAM/GPU | Total VRAM | Interconnect |
|-------|------|----------|----------|------------|--------------|
| Small (≤1B) | 1-8 | A100-40GB | 40GB | 40-320GB | PCIe OK |
| Medium (3-8B) | 8-16 | A100-40GB | 40GB | 320-640GB | NVLink preferred |
| Large (12-24B) | 32-64 | A100-80GB | 80GB | 2.5-5TB | NVLink required |
| XL (48-100B) | 128-512 | H100 | 80GB | 10-40TB | NVSwitch required |

### 10.2 Storage Requirements

```
┌─────────────────────────────────────────────────────────────────┐
│         STORAGE REQUIREMENTS (Budget-Constrained)               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Training Data (Modal Volume):                                  │
│  ├─ FineWeb-Edu (sample):   10 GB                              │
│  ├─ Code datasets:          5 GB                               │
│  ├─ Math/Science:           2 GB                               │
│  └─ Tokenized cache:        ~20 GB                             │
│                                                                 │
│  Checkpoints (per backend):                                     │
│  ├─ TINY:    20 MB × 2 = 40 MB                                 │
│  ├─ 256M:    0.5 GB × 4 = 2 GB                                 │
│  ├─ 512M:    1 GB × 6 = 6 GB                                   │
│  └─ Total per backend: ~8 GB                                   │
│                                                                 │
│  Logs/Artifacts (per backend):                                  │
│  ├─ TensorBoard:            500 MB per run                     │
│  ├─ W&B (local):            500 MB per run                     │
│  └─ Profiler traces:        200 MB per run                     │
│                                                                 │
│  TOTAL (both backends):     ~50 GB                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 10.3 Network Requirements (Modal)

| Configuration | Bandwidth (inter-GPU) | Latency | Notes |
|---------------|----------------------|---------|-------|
| Modal 8×A100-80GB | NVLink | <1 μs | Optimal for our setup |
| GPU-to-Storage | Modal internal | ~10ms | Volume access |
| External egress | Variable | N/A | Minimal (local W&B) |

### 10.4 Software Dependencies

```toml
# pyproject.toml dependencies
[project]
dependencies = [
    # Core
    "torch>=2.1.0",
    "transformers>=4.35.0",
    "datasets>=2.14.0",
    "accelerate>=0.24.0",

    # Distributed
    "ray[train,data]>=2.9.0",
    "deepspeed>=0.12.0",

    # Monitoring (local mode)
    "wandb>=0.16.0",
    "tensorboard>=2.15.0",
    "rich>=13.0.0",

    # Data
    "safetensors>=0.4.0",
    "pyarrow>=14.0.0",

    # Cloud
    "modal>=0.55.0",
]

[project.optional-dependencies]
rust = ["maturin>=1.4.0", "candle-core>=0.3.0"]
```

---

## 11. Risk Mitigation Strategies

### 11.1 Risk Matrix (Budget: $500 per Backend)

| Risk | Probability | Impact | Mitigation | Owner Task |
|------|-------------|--------|------------|------------|
| **Budget Overrun** | High | Critical | $500 hard limit per backend, alerts at $250/$375/$450/$475 | R1 |
| **Training Divergence** | Medium | High | Gradient clipping, LR warmup, checkpoint every 500 steps | R2 |
| **OOM Errors** | Medium | High | Gradient checkpointing, batch size auto-tuning | R3 |
| **Backend Failure** | Medium | Medium | Automatic fallback, checkpoint recovery | R4 |
| **Data Quality Issues** | Low | Medium | Validation split monitoring, perplexity tracking | R5 |
| **Slow Convergence** | Medium | Medium | LR schedule tuning, early stopping criteria | R6 |
| **Logging Failure** | Low | Medium | Local W&B backup, TensorBoard redundancy | R7 |

### 11.2 Risk Mitigation Task Checklist

```
┌─────────────────────────────────────────────────────────────────┐
│              RISK MITIGATION TASKS                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  R1: BUDGET OVERRUN PREVENTION                                  │
│  ─────────────────────────────────────────────────────────────  │
│  □ R1.1 Configure $500 hard limit per backend in cost_tracker  │
│  □ R1.2 Set up alert callbacks at $250 (50%)                   │
│  □ R1.3 Set up alert callbacks at $375 (75%)                   │
│  □ R1.4 Set up alert callbacks at $450 (90%)                   │
│  □ R1.5 Set up alert callbacks at $475 (95%)                   │
│  □ R1.6 Implement auto-checkpoint on $475 threshold            │
│  □ R1.7 Implement auto-stop on $495 threshold                  │
│  □ R1.8 Create budget dashboard in TensorBoard                 │
│                                                                 │
│  R2: TRAINING DIVERGENCE PREVENTION                             │
│  ─────────────────────────────────────────────────────────────  │
│  □ R2.1 Configure gradient clipping (max_norm=1.0)             │
│  □ R2.2 Set up loss spike detection (>3σ from rolling mean)    │
│  □ R2.3 Implement automatic LR reduction on spike              │
│  □ R2.4 Configure warmup steps (500 for 256M, 1000 for 512M)   │
│  □ R2.5 Enable NaN/Inf detection in loss                       │
│  □ R2.6 Set up automatic rollback to last good checkpoint      │
│                                                                 │
│  R3: OOM PREVENTION                                             │
│  ─────────────────────────────────────────────────────────────  │
│  □ R3.1 Enable gradient checkpointing by default               │
│  □ R3.2 Configure batch size auto-scaling                      │
│  □ R3.3 Set memory monitoring alerts at 90% VRAM               │
│  □ R3.4 Test memory usage on TINY before larger runs           │
│  □ R3.5 Document fallback batch sizes per model                │
│                                                                 │
│  R4: BACKEND FAILURE RECOVERY                                   │
│  ─────────────────────────────────────────────────────────────  │
│  □ R4.1 Implement checkpoint save on SIGTERM                   │
│  □ R4.2 Configure automatic retry (max 3 attempts)             │
│  □ R4.3 Set up resume-from-checkpoint logic                    │
│  □ R4.4 Create backend health check endpoint                   │
│  □ R4.5 Configure Modal container restart policy               │
│                                                                 │
│  R5: DATA QUALITY MONITORING                                    │
│  ─────────────────────────────────────────────────────────────  │
│  □ R5.1 Set up validation loss tracking every 100 steps        │
│  □ R5.2 Configure perplexity monitoring                        │
│  □ R5.3 Implement data sample logging to W&B                   │
│  □ R5.4 Set up anomaly detection for loss curves               │
│                                                                 │
│  R6: CONVERGENCE OPTIMIZATION                                   │
│  ─────────────────────────────────────────────────────────────  │
│  □ R6.1 Configure cosine LR schedule with warmup               │
│  □ R6.2 Set up early stopping (patience=5 evals)               │
│  □ R6.3 Implement learning rate finder (optional)              │
│  □ R6.4 Document expected loss curves per model size           │
│                                                                 │
│  R7: LOGGING REDUNDANCY                                         │
│  ─────────────────────────────────────────────────────────────  │
│  □ R7.1 Configure W&B offline mode with local sync             │
│  □ R7.2 Set up TensorBoard as backup logging                   │
│  □ R7.3 Implement JSON log fallback                            │
│  □ R7.4 Configure artifact backup to Modal volume              │
│  □ R7.5 Set up periodic log sync (every 100 steps)             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 11.3 Checkpoint Recovery Protocol

```python
# Automatic checkpoint recovery in distributed_trainer.py
CheckpointRecovery:
    # On failure:
    1. Save emergency checkpoint (current state)
    2. Log failure with stack trace to W&B local
    3. Retry from last successful checkpoint
    4. If 3 consecutive failures → alert and pause

    # On preemption:
    1. Receive SIGTERM signal
    2. Save checkpoint within 30-second grace period
    3. Exit cleanly for retry

    # Recovery command (Rust+GPU):
    uv run modal run ray_cluster.py::run_rust \
        --resume-from /vol/checkpoints/rust/step_5000.pt

    # Recovery command (PyTorch+GPU):
    uv run modal run ray_cluster.py::run_pytorch \
        --resume-from /vol/checkpoints/pytorch/step_5000.pt
```

### 11.4 Budget Alert System ($500 per Backend)

```python
# Budget Alert Configuration for Dual Backend Training
BUDGET_ALERT_CONFIG = {
    "rust_gpu": {
        "budget_limit": 500.0,          # USD
        "thresholds": {
            250.0: {                      # 50%
                "level": "INFO",
                "message": "Rust+GPU: 50% budget used ($250). On track.",
                "action": "log_to_wandb",
            },
            375.0: {                      # 75%
                "level": "WARNING",
                "message": "Rust+GPU: 75% budget used ($375). Review progress.",
                "action": "log_to_wandb",
            },
            450.0: {                      # 90%
                "level": "CRITICAL",
                "message": "Rust+GPU: 90% budget used ($450). Consider pausing.",
                "action": "save_checkpoint",
            },
            475.0: {                      # 95%
                "level": "EXCEEDED",
                "message": "Rust+GPU: 95% budget used ($475). Auto-checkpoint triggered.",
                "action": "save_checkpoint_and_alert",
            },
            495.0: {                      # 99%
                "level": "STOP",
                "message": "Rust+GPU: 99% budget used ($495). Stopping training.",
                "action": "stop_training",
            },
        },
    },
    "pytorch_gpu": {
        "budget_limit": 500.0,          # USD
        "thresholds": {
            250.0: {                      # 50%
                "level": "INFO",
                "message": "PyTorch+GPU: 50% budget used ($250). On track.",
                "action": "log_to_wandb",
            },
            375.0: {                      # 75%
                "level": "WARNING",
                "message": "PyTorch+GPU: 75% budget used ($375). Review progress.",
                "action": "log_to_wandb",
            },
            450.0: {                      # 90%
                "level": "CRITICAL",
                "message": "PyTorch+GPU: 90% budget used ($450). Consider pausing.",
                "action": "save_checkpoint",
            },
            475.0: {                      # 95%
                "level": "EXCEEDED",
                "message": "PyTorch+GPU: 95% budget used ($475). Auto-checkpoint triggered.",
                "action": "save_checkpoint_and_alert",
            },
            495.0: {                      # 99%
                "level": "STOP",
                "message": "PyTorch+GPU: 99% budget used ($495). Stopping training.",
                "action": "stop_training",
            },
        },
    },
}

# Alert Actions Implementation
def setup_budget_alerts(backend: str, tracker: CostTracker):
    config = BUDGET_ALERT_CONFIG[backend]
    
    for threshold, alert in config["thresholds"].items():
        tracker.add_alert_callback(
            threshold=threshold,
            level=alert["level"],
            callback=lambda a: execute_alert_action(a, alert["action"])
        )

def execute_alert_action(alert, action: str):
    if action == "log_to_wandb":
        wandb.log({"budget_alert": alert.message})
    elif action == "save_checkpoint":
        save_emergency_checkpoint()
        wandb.log({"budget_alert": alert.message, "checkpoint_saved": True})
    elif action == "save_checkpoint_and_alert":
        save_emergency_checkpoint()
        send_notification(alert.message)  # Slack/Email
    elif action == "stop_training":
        save_emergency_checkpoint()
        raise BudgetExhaustedError(alert.message)
```

### 11.5 Fallback Strategies ($500 Budget Context)

```
┌─────────────────────────────────────────────────────────────────┐
│              FALLBACK DECISION TREE ($500/backend)              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Training not converging?                                       │
│  ├─ Loss stuck → Reduce LR by 50%, extend warmup               │
│  ├─ Loss spiking → Reduce batch size, increase grad clip       │
│  └─ NaN loss → Rollback to last checkpoint, reduce LR to 1e-5  │
│                                                                 │
│  OOM errors?                                                    │
│  ├─ Enable gradient checkpointing (first try)                  │
│  ├─ Reduce batch size by 50%                                   │
│  ├─ Reduce context length (512 → 256)                         │
│  └─ Switch to smaller model size                               │
│                                                                 │
│  Budget running low (<$100 remaining)?                          │
│  ├─ Focus on smaller model (256M instead of 512M)              │
│  ├─ Reduce training steps proportionally                       │
│  ├─ Skip non-essential ablations                               │
│  └─ Save final checkpoint immediately                          │
│                                                                 │
│  Backend comparison imbalanced?                                 │
│  ├─ One backend significantly faster → allocate more budget    │
│  ├─ One backend failing → focus resources on working backend   │
│  └─ Both performing similar → proceed with equal allocation    │
│                                                                 │
│  Logging failure?                                               │
│  ├─ W&B failing → rely on TensorBoard                         │
│  ├─ TensorBoard failing → rely on JSON logs                   │
│  └─ All logging failing → continue with console output         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```
│  └─ Use Flash Attention if available                           │
│                                                                 │
│  Budget issues?                                                 │
│  ├─ Switch to spot instances (60-70% savings)                  │
│  ├─ Reduce model size (train smaller prototype first)          │
│  ├─ Reduce training tokens (may impact quality)                │
│  └─ Pause and resume during off-peak hours                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 12. Appendices

### Appendix A: Command Reference

#### A.1 Training Commands

```bash
# Single-GPU tiny model (development)
uv run python scripts/train_tiny.py \
    --model-size tiny \
    --max-steps 1000 \
    --output-dir ./outputs/tiny

# 8-GPU distributed training (Modal)
uv run modal run src/deepseek/cloud/modal/ray_cluster.py::run_pytorch \
    --scale initial \
    --model-size 1B \
    --max-steps 25000 \
    --checkpoint-dir /vol/checkpoints

# 64-GPU scaled training (Modal)
uv run modal run src/deepseek/cloud/modal/ray_cluster.py::run_pytorch \
    --scale scaled \
    --model-size 8B \
    --max-steps 100000 \
    --fsdp-sharding FULL_SHARD

# Rust backend training (verification)
uv run modal run src/deepseek/cloud/modal/ray_cluster.py::run_rust \
    --scale initial \
    --max-steps 100
```

#### A.2 Evaluation Commands

```bash
# Perplexity evaluation
uv run python scripts/evaluate.py \
    --checkpoint ./outputs/1B/checkpoint_final \
    --eval-data ./data/eval \
    --metrics perplexity

# Downstream evaluation
uv run python scripts/downstream_eval.py \
    --checkpoint ./outputs/1B/checkpoint_final \
    --tasks hellaswag arc_easy arc_challenge winogrande

# Benchmark throughput
uv run python scripts/benchmark.py \
    --checkpoint ./outputs/1B/checkpoint_final \
    --batch-sizes 1 4 8 16 \
    --seq-lengths 128 512 2048
```

#### A.3 Export Commands

```bash
# Export to SafeTensors
uv run python scripts/export_gguf.py \
    --checkpoint ./outputs/1B/checkpoint_final \
    --output ./exports/model.safetensors \
    --format safetensors

# Export to GGUF (for llama.cpp)
uv run python scripts/export_gguf.py \
    --checkpoint ./outputs/1B/checkpoint_final \
    --output ./exports/model.gguf \
    --quantize q4_0
```

### Appendix B: Configuration Templates

#### B.1 Small Model Config (1B)

```python
# config/1B_config.py
from deepseek.pipeline.config import PipelineConfig, ModelConfig, TrainingConfig

config = PipelineConfig(
    model=ModelConfig(
        d_model=1536,
        num_heads=16,
        num_layers=24,
        intermediate_size=6144,
        vocab_size=100000,
        max_seq_len=4096,
        use_mla=True,
        d_latent=512,
        use_moe=False,
    ),
    training=TrainingConfig(
        learning_rate=3e-4,
        batch_size=8,
        gradient_accumulation_steps=4,
        max_steps=25000,
        warmup_steps=1000,
        weight_decay=0.1,
        max_grad_norm=1.0,
        use_amp=True,
        save_every_n_steps=500,
    ),
    data=DataConfig(
        domain_weights={"web": 0.7, "code": 0.15, "math": 0.08, "papers": 0.04, "scientific": 0.03},
        max_seq_length=4096,
    ),
)
```

#### B.2 Medium Model Config (8B)

```python
# config/8B_config.py
config = PipelineConfig(
    model=ModelConfig(
        d_model=4096,
        num_heads=32,
        num_layers=32,
        intermediate_size=14336,
        vocab_size=100000,
        max_seq_len=8192,
        use_mla=True,
        d_latent=768,
        use_moe=True,
        num_experts=16,
        top_k=2,
    ),
    training=TrainingConfig(
        learning_rate=1.5e-4,
        batch_size=4,
        gradient_accumulation_steps=8,
        max_steps=100000,
        warmup_steps=3000,
        scheduler="warmup_stable_decay",
        use_amp=True,
        use_gradient_checkpointing=True,
    ),
    distributed=DistributedConfig(
        num_workers=8,
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        data_parallel_size=2,
    ),
)
```

### Appendix C: Monitoring Dashboard (W&B Local + TensorBoard)

```python
# Launch real-time dashboard with W&B local and TensorBoard
import wandb
from torch.utils.tensorboard import SummaryWriter
from monitoring.dashboard import TrainingDashboard
from monitoring.cost_tracker import CostTracker

# Setup W&B in offline mode (local)
wandb.init(
    mode="offline",
    project="deepseek-training",
    dir="/vol/wandb",
    config={
        "backend": "rust_gpu | pytorch_gpu",
        "budget_limit": 500.0,
        "platform": "modal",
        "gpu_config": "A100-80GB x 8",
    }
)

# Setup TensorBoard
tb_writer = SummaryWriter(log_dir="/vol/tensorboard/{backend}")

# Initialize cost tracker with $500 budget per backend
tracker = CostTracker(budget_limit=500.0)
dashboard = TrainingDashboard(cost_tracker=tracker)

# Setup budget alerts ($500 per backend)
tracker.add_alert_callback(threshold=250.0, callback=lambda a: wandb.log({"alert": "50%"}))
tracker.add_alert_callback(threshold=375.0, callback=lambda a: wandb.log({"alert": "75%"}))
tracker.add_alert_callback(threshold=450.0, callback=lambda a: save_checkpoint())
tracker.add_alert_callback(threshold=475.0, callback=lambda a: send_alert_and_checkpoint())

# Start dashboard
dashboard.start(total_steps=10000)

# Training loop with dual logging
for step in range(10000):
    loss = train_step()
    
    # Log to W&B (local)
    wandb.log({
        "loss": loss,
        "learning_rate": scheduler.get_lr(),
        "tokens_per_second": throughput,
        "cost_usd": tracker.get_current_cost(),
    })
    
    # Log to TensorBoard
    tb_writer.add_scalar("Loss/train", loss, step)
    tb_writer.add_scalar("LR", scheduler.get_lr(), step)
    tb_writer.add_scalar("Budget/cost_usd", tracker.get_current_cost(), step)
    
    dashboard.update(
        step=step,
        loss=loss,
        learning_rate=scheduler.get_lr(),
        tokens_per_second=throughput,
        gpu_memory_percent=memory_usage,
    )

dashboard.stop()
```

### Appendix D: Research References

1. **Chinchilla Scaling Laws**: [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) - Hoffmann et al., 2022
2. **DeepSeek-V3 Technical Report**: [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) - DeepSeek-AI, 2024
3. **Llama 3 Paper**: [Meta AI Blog](https://ai.meta.com/blog/meta-llama-3/)
4. **FlashAttention**: [FlashAttention-2](https://arxiv.org/abs/2307.08691) - Dao, 2023
5. **ZeRO Optimization**: [ZeRO: Memory Optimizations](https://arxiv.org/abs/1910.02054) - Rajbhandari et al., 2019
6. **Mixture of Experts**: [Switch Transformers](https://arxiv.org/abs/2101.03961) - Fedus et al., 2021

---

## Document Metadata

| Field | Value |
|-------|-------|
| **Version** | 2.0 |
| **Status** | Complete |
| **Last Updated** | December 2025 |
| **Platform** | Modal (A100-80GB × 8) |
| **Backends** | Rust+GPU, PyTorch+GPU |
| **Budget** | $500 per backend ($1,000 total) |
| **Logging** | W&B (local) + TensorBoard |
| **Author** | AI-Generated Training Plan |
| **Review Status** | Pending |

---

*This document was generated for Modal-exclusive training with A100-80GB × 8 configuration. The plan compares Rust+GPU vs PyTorch+GPU backends with a $500 budget limit per backend. All logging uses W&B in local/offline mode and TensorBoard for redundancy. Cost estimates are based on Modal's December 2025 pricing ($2.78/hr per A100-80GB).*
