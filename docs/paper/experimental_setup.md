# Experimental Setup

This document describes the experimental setup used for training and evaluating our DeepSeek-V3 implementation.

## Hardware Configuration

### Primary Training Setup

| Component | Specification |
|-----------|--------------|
| GPUs | 3× NVIDIA H100 80GB |
| CPU | AMD EPYC 7763 (64 cores) |
| RAM | 512 GB DDR4-3200 |
| Storage | 4TB NVMe SSD |
| Interconnect | NVLink 4.0 (900 GB/s) |

### Alternative Configurations

#### Apple Silicon (Local Development)
| Component | Specification |
|-----------|--------------|
| Chip | Apple M1/M2/M3 Pro/Max |
| Unified Memory | 16-64 GB |
| Storage | 512GB - 2TB SSD |

#### Cloud GPUs (Modal)
| Instance | GPUs | VRAM | Cost/hr |
|----------|------|------|---------|
| A10G | 1× | 24 GB | $1.10 |
| A100-40 | 1× | 40 GB | $3.20 |
| A100-80 | 1× | 80 GB | $4.25 |
| H100 | 1× | 80 GB | $5.50 |

## Software Environment

### Python Dependencies

```
Python: 3.12.x
PyTorch: 2.2.0+
CUDA: 12.1
cuDNN: 8.9.x

# Key packages
torch>=2.2.0
mlx>=0.5.0
transformers>=4.36.0
einops>=0.7.0
wandb>=0.16.0
hydra-core>=1.3.0
ray>=2.9.0
```

### Rust Dependencies

```toml
[dependencies]
candle-core = "0.4"
candle-nn = "0.4"
candle-transformers = "0.4"
tokenizers = "0.15"
```

## Model Configurations

### Tiny Model (Default for Ablations)

```yaml
model:
  d_model: 256
  n_layers: 6
  n_heads: 8
  d_head: 32
  d_ff: 1024
  vocab_size: 32000
  
  # MLA
  d_latent_kv: 64
  d_latent_q: 128
  d_rope: 16
  
  # MoE
  num_experts: 8
  num_shared_experts: 1
  top_k: 2
  expert_dim: 256
  
  # MTP
  mtp_depth: 1
  mtp_weight: 0.3

total_params: ~10M
active_params: ~8M
```

### Small Model

```yaml
model:
  d_model: 512
  n_layers: 12
  n_heads: 16
  d_head: 32
  d_ff: 2048
  vocab_size: 32000
  
  # MoE
  num_experts: 64
  top_k: 4

total_params: ~100M
active_params: ~50M
```

### Medium Model

```yaml
model:
  d_model: 1024
  n_layers: 24
  n_heads: 32
  d_head: 32
  d_ff: 4096
  vocab_size: 32000
  
  # MoE
  num_experts: 256
  top_k: 8

total_params: ~1B
active_params: ~200M
```

## Training Configuration

### Optimization

```yaml
optimizer:
  type: AdamW
  lr: 3e-4
  betas: [0.9, 0.95]
  weight_decay: 0.1
  eps: 1e-8

scheduler:
  type: cosine
  warmup_steps: 1000
  min_lr_ratio: 0.1

gradient:
  clip_norm: 1.0
  accumulation_steps: 4
```

### Mixed Precision

```yaml
precision:
  compute: bfloat16
  accumulation: float32
  master_weights: float32
  
# For FP8 ablations
fp8:
  enabled: true
  format: e4m3
  block_size: 128
```

### Distributed Training

```yaml
distributed:
  strategy: fsdp  # or ddp
  
  # FSDP specific
  fsdp:
    sharding_strategy: FULL_SHARD
    backward_prefetch: BACKWARD_PRE
    cpu_offload: false
    
  # Pipeline parallelism
  pipeline:
    enabled: true
    stages: 3
    micro_batches: 8
    schedule: dualpipe  # or 1f1b
```

## Dataset

### TinyStories (Primary)

```
Dataset: roneneldan/TinyStories
Size: ~2.5M stories
Tokens: ~500M tokens
Vocabulary: GPT-2 tokenizer (50257 tokens)
Split: 95% train, 5% validation
```

### Data Processing

```python
# Tokenization
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Processing
max_length = 512
batch_size = 32  # per GPU
sequence_packing = True
```

## Evaluation Metrics

### Primary Metrics

1. **Validation Loss**: Cross-entropy on held-out data
2. **Perplexity**: exp(validation_loss)
3. **Throughput**: Tokens processed per second
4. **Memory**: Peak GPU memory usage

### Secondary Metrics

1. **Expert Load Balance**: Coefficient of Variation across experts
2. **Gradient Norm**: L2 norm of gradients
3. **Learning Rate**: Current LR from scheduler
4. **MTP Accuracy**: Accuracy of auxiliary predictions

## Ablation Protocol

### Seeds

All experiments run with 3 seeds:
- Seed 42 (primary)
- Seed 123 (secondary)
- Seed 456 (tertiary)

### Statistical Analysis

```python
# Report format
mean = np.mean(results)
std = np.std(results)
ci_95 = 1.96 * std / np.sqrt(n_seeds)
print(f"{mean:.4f} ± {ci_95:.4f}")

# Significance testing
from scipy.stats import ttest_ind
t_stat, p_value = ttest_ind(baseline, experimental)
significant = p_value < 0.05
```

### Compute Budget

| Experiment | Steps | Time (H100) | Cost |
|------------|-------|-------------|------|
| Single ablation | 1000 | ~10 min | ~$1 |
| Full ablation (3 seeds) | 3000 | ~30 min | ~$3 |
| Benchmark comparison | 5000 | ~50 min | ~$5 |
| Full training | 50000 | ~8 hr | ~$45 |

## Reproducibility Checklist

- [ ] Fixed random seeds (PyTorch, NumPy, Python)
- [ ] Deterministic operations where possible
- [ ] Exact software versions logged
- [ ] Data preprocessing documented
- [ ] Model initialization documented
- [ ] Hyperparameters logged to W&B
- [ ] Checkpoints saved at regular intervals
- [ ] Hardware configuration recorded

## Experiment Tracking

### Weights & Biases Configuration

```yaml
wandb:
  project: deepseek-from-scratch
  entity: dev-jadhav
  
  log:
    - train/loss
    - train/lr
    - val/loss
    - val/perplexity
    - system/gpu_memory
    - system/throughput
    - moe/expert_load_cv
    - moe/routing_entropy
    
  save:
    - model_config
    - training_config
    - checkpoints (every 1000 steps)
```

### Logging Frequency

| Metric Type | Frequency |
|-------------|-----------|
| Training loss | Every step |
| Learning rate | Every step |
| Gradient norm | Every 10 steps |
| Validation metrics | Every 100 steps |
| Expert statistics | Every 100 steps |
| Checkpoints | Every 1000 steps |

## Environment Variables

```bash
# CUDA
export CUDA_VISIBLE_DEVICES=0,1,2
export NCCL_DEBUG=INFO

# PyTorch
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Reproducibility
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42

# W&B
export WANDB_PROJECT=deepseek-from-scratch
export WANDB_MODE=online
```
